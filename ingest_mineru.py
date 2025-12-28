import re
import json
import uuid
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.types import Metadata
from sentence_transformers import SentenceTransformer
import tantivy

PDF_DIR = Path("data/pdfs")
MINERU_OUT = Path("data/mineru_out")
CHROMA_DIR = Path("chroma")
TANTIVY_DIR = Path("tantivy_index")

COLLECTION_NAME = "pdf_chunks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_CHARS = 1400
CHUNK_OVERLAP = 200


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return out


def run_mineru(pdf_path: Path, out_root: Path) -> Path:
    """
    Extract PDF into out_root/<pdf_stem>/... (markdown files).
    Adjust the command to match your MinerU installation.
    """
    pdf_out = out_root / pdf_path.stem
    pdf_out.mkdir(parents=True, exist_ok=True)

    # If you're using Docker wrapper:
    # cmd = ["./mineru.sh", str(pdf_path), str(pdf_out)]
    # subprocess.run(cmd, check=True)

    # If MinerU is installed locally:
    cmd = ["mineru", "-p", str(pdf_path), "-o", str(pdf_out)]
    subprocess.run(cmd, check=True)

    return pdf_out



def load_extracted_markdown(pdf_stem_dir: Path) -> str:
    """Fallback: concatenate MinerU-produced markdown files."""
    md_files = sorted(pdf_stem_dir.glob("**/*.md"))
    if not md_files:
        return ""
    parts = []
    for f in md_files:
        parts.append(f.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(parts)


def _as_int_page(val: Any) -> int:
    """Best-effort page normalization."""
    try:
        n = int(float(val))
        return n
    except Exception:
        return 0


def _maybe_heading_text(obj: Any) -> str:
    """Try to identify heading/section titles from a block-like object."""
    if not isinstance(obj, dict):
        return ""

    t = str(obj.get("type", "") or obj.get("block_type", "") or "").lower()
    if t in {"title", "heading", "header", "section", "chapter", "subtitle"}:
        txt = obj.get("text") or obj.get("content") or obj.get("md") or obj.get("markdown")
        return clean_text(str(txt)) if txt else ""

    if obj.get("is_title") is True or obj.get("is_heading") is True:
        txt = obj.get("text") or obj.get("content") or obj.get("md") or obj.get("markdown")
        return clean_text(str(txt)) if txt else ""

    return ""


def _extract_text_from_block(obj: Any) -> str:
    """Extract textual content from a block-like object."""
    if isinstance(obj, str):
        return clean_text(obj)
    if not isinstance(obj, dict):
        return ""

    # Prefer structured line lists when available; MinerU often stores richer
    # paragraph content in `lines` while `text` can be line-fragmentary.
    if isinstance(obj.get("lines"), list):
        lines = []
        for ln in obj["lines"]:
            if isinstance(ln, dict) and isinstance(ln.get("text"), str):
                lines.append(ln["text"])
            elif isinstance(ln, str):
                lines.append(ln)
        joined = clean_text("\n".join(lines))
        if joined:
            return joined

    for k in ("text", "content", "md", "markdown", "paragraph"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return clean_text(v)

    return ""


def _iter_json_blocks(obj: Any, page: int = 0, section: str = ""):
    """Traverse arbitrary MinerU JSON and yield (text, page, section)."""
    if isinstance(obj, dict):
        # Update page context if present
        for pk in ("page", "page_no", "page_num", "page_index", "pageNumber"):
            if pk in obj and obj.get(pk) is not None:
                page = _as_int_page(obj.get(pk))
                break

        # Update section context if this looks like a heading
        heading = _maybe_heading_text(obj)
        if heading:
            section = heading

        # If this dict itself is a text block, yield it
        txt = _extract_text_from_block(obj)
        if txt:
            yield (txt, page, section)

        # Recurse into likely containers
        for key in ("pages", "blocks", "elements", "content", "children", "items", "paragraphs"):
            v = obj.get(key)
            if isinstance(v, list):
                for it in v:
                    yield from _iter_json_blocks(it, page=page, section=section)
            elif isinstance(v, dict):
                yield from _iter_json_blocks(v, page=page, section=section)

        # Defensive recursion into other nested structures
        for k, v in obj.items():
            if k in {"pages", "blocks", "elements", "content", "children", "items", "paragraphs", "lines"}:
                continue
            if isinstance(v, (dict, list)):
                yield from _iter_json_blocks(v, page=page, section=section)

    elif isinstance(obj, list):
        for it in obj:
            yield from _iter_json_blocks(it, page=page, section=section)


def load_extracted_json_blocks(pdf_stem_dir: Path) -> List[Dict[str, Any]]:
    """Prefer MinerU JSON output. Returns list of blocks with text + (page, section)."""
    json_files = sorted(
        f for f in pdf_stem_dir.glob("**/*.json")
        if f.name not in {"meta.json", "metadata.json"}
    )
    if not json_files:
        return []

    blocks: List[Dict[str, Any]] = []
    for jf in json_files:
        try:
            obj = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        for text, page, section in _iter_json_blocks(obj, page=0, section=""):
            if not text:
                continue
            blocks.append({"text": text, "page": page, "section": section})

    # De-duplicate exact repeats
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for b in blocks:
        key = (b.get("page", 0), b.get("section", ""), b.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)

    return deduped


def build_tantivy() -> Tuple[tantivy.Index, tantivy.IndexWriter]:
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("id", stored=True)
    schema_builder.add_text_field("content", stored=True)
    schema_builder.add_text_field("source_file", stored=True)
    schema_builder.add_integer_field("page", stored=True, indexed=True)
    schema_builder.add_text_field("section", stored=True)
    schema = schema_builder.build()

    TANTIVY_DIR.mkdir(parents=True, exist_ok=True)
    if (TANTIVY_DIR / "meta.json").exists():
        index = tantivy.Index.open(str(TANTIVY_DIR))
    else:
        index = tantivy.Index(schema, path=str(TANTIVY_DIR))
    writer = index.writer(50_000_000)
    return index, writer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest PDFs via MinerU into Chroma + Tantivy.")
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild local indexes (chroma/ and tantivy_index/). Requires --yes.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive operations when used with --reset.",
    )
    return p.parse_args()


def _reset_indexes(yes: bool) -> None:
    if not yes:
        raise SystemExit(
            "Refusing to reset without confirmation. Re-run with: python ingest_mineru.py --reset --yes"
        )

    for d in (CHROMA_DIR, TANTIVY_DIR):
        if d.exists():
            shutil.rmtree(d)


def main(reset: bool = False, yes: bool = False) -> None:
    if reset:
        _reset_indexes(yes=yes)

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    MINERU_OUT.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR.resolve()}")
        return

    # Vector store
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

    embedder = SentenceTransformer(EMBED_MODEL)

    # BM25
    bm25_index, bm25_writer = build_tantivy()

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Metadata] = []

    for pdf_path in pdfs:
        pdf_out = MINERU_OUT / pdf_path.stem
        has_json = pdf_out.exists() and any(
            f.name not in {"meta.json", "metadata.json"}
            for f in pdf_out.glob("**/*.json")
        )
        has_md = pdf_out.exists() and any(True for _ in pdf_out.glob("**/*.md"))

        if not pdf_out.exists() or (not has_json and not has_md):
            print(f"[MinerU] Extracting {pdf_path.name} ...")
            pdf_out = run_mineru(pdf_path, MINERU_OUT)
            has_json = any(
                f.name not in {"meta.json", "metadata.json"}
                for f in pdf_out.glob("**/*.json")
            )
            has_md = any(True for _ in pdf_out.glob("**/*.md"))

        # Prefer JSON blocks (better metadata), fall back to Markdown
        json_blocks = load_extracted_json_blocks(pdf_out) if has_json else []

        if json_blocks:
            # MinerU JSON can be very granular (often line fragments). Merge consecutive
            # fragments per (page, section) so we index paragraph-like blocks.
            merged_blocks: List[Dict[str, Any]] = []
            merge_target_chars = 4000

            cur_page = None
            cur_section = None
            cur_parts: List[str] = []
            cur_len = 0

            def _flush_current() -> None:
                nonlocal cur_page, cur_section, cur_parts, cur_len
                if cur_parts:
                    merged_blocks.append(
                        {
                            "text": clean_text("\n".join(cur_parts)),
                            "page": int(cur_page or 0),
                            "section": str(cur_section or ""),
                        }
                    )
                cur_page = None
                cur_section = None
                cur_parts = []
                cur_len = 0

            chunk_count = 0
            for block in json_blocks:
                text = block.get("text", "")
                if not text.strip():
                    continue

                page_val = int(block.get("page", 0) or 0)
                section_val = str(block.get("section", "") or "")
                text_clean = clean_text(text)
                if not text_clean:
                    continue

                if cur_page is None:
                    cur_page = page_val
                    cur_section = section_val

                # Start a new merged block if (page, section) changes or we hit target size.
                if (page_val != cur_page) or (section_val != cur_section) or (cur_len >= merge_target_chars):
                    _flush_current()
                    cur_page = page_val
                    cur_section = section_val

                cur_parts.append(text_clean)
                cur_len += len(text_clean)

            _flush_current()

            for block in merged_blocks:
                text = block.get("text", "")
                if not text.strip():
                    continue

                block_chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
                for chunk in block_chunks:
                    _id = str(uuid.uuid4())
                    meta = {
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "page": int(block.get("page", 0) or 0),
                        "section": str(block.get("section", "") or ""),
                        "chunk_index": chunk_count,
                    }
                    ids.append(_id)
                    docs.append(chunk)
                    metas.append(meta)
                    chunk_count += 1

            print(f"[Chunk] {pdf_path.name}: {chunk_count} chunks (from JSON)")

        else:
            md_text = load_extracted_markdown(pdf_out) if has_md else ""
            if not md_text.strip():
                print(f"WARNING: MinerU produced no JSON/markdown for {pdf_path.name}")
                continue

            chunks = chunk_text(md_text, CHUNK_CHARS, CHUNK_OVERLAP)
            print(f"[Chunk] {pdf_path.name}: {len(chunks)} chunks (from Markdown)")

            for i, chunk in enumerate(chunks):
                _id = str(uuid.uuid4())
                meta = {
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page": 0,
                    "section": "",
                    "chunk_index": i,
                }
                ids.append(_id)
                docs.append(chunk)
                metas.append(meta)

    if not docs:
        print("No chunks to index. Exiting.")
        return

    print(f"[Embed] Embedding {len(docs)} chunks...")
    vectors = embedder.encode(docs, show_progress_bar=True).tolist()

    print("[Chroma] Upserting...")
    # ChromaDB has a max batch size of ~5461; batch to avoid overflow
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        batch_ids = ids[i:batch_end]
        batch_docs = docs[i:batch_end]
        batch_metas = metas[i:batch_end]
        batch_vecs = vectors[i:batch_end]
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_vecs)
        print(f"  Upserted {batch_end}/{len(ids)} chunks")

    print("[BM25] Indexing Tantivy...")
    for _id, content, meta in zip(ids, docs, metas):
        page_val = meta.get("page", 0)
        section_val = meta.get("section", "")
        # Ensure page is an unsigned integer (U64) by using abs() and ensuring non-negative
        page_int = abs(int(page_val)) if isinstance(page_val, (int, float, str)) else 0
        bm25_writer.add_document(tantivy.Document(
            id=_id,
            content=content,
            source_file=str(meta["source_file"]),
            page=page_int,
            section=str(section_val) if section_val else "",
        ))
    bm25_writer.commit()

    print("[Done] Indexed successfully.")
    print(f"  Chroma collection: {COLLECTION_NAME}")
    print(f"  Tantivy index dir: {TANTIVY_DIR.resolve()}")


if __name__ == "__main__":
    args = _parse_args()
    main(reset=args.reset, yes=args.yes)