# Retrieval-Constrained Misconception Diagnosis for Short Math Explanations: A System Engineering Study on a Closed-World MAP Benchmark

**Anonymous authors**

## Abstract
Misconception diagnosis is a distinct assessment task from grading: it aims to infer a learner’s underlying (often coherent) mental model that produces systematic errors, not merely to mark an answer as correct or incorrect. In educational settings, hallucinated rationales or ungrounded labels are unacceptable because they can mislead learners and educators and undermine accountability. This paper describes a retrieval-constrained large language model (LLM) system that performs closed-world misconception diagnosis for short, free-text mathematics explanations. The system is engineered to (i) retrieve evidence first, (ii) construct a bounded hypothesis space of candidate misconception labels derived only from retrieved evidence, and (iii) constrain the LLM to rank and select among these candidates with a fixed schema. A secondary LLM operates as a reliability component to audit schema compliance, candidate-set adherence, grounding in retrieved evidence, and stability across reruns. Evaluation on a closed-world MAP benchmark with a fixed misconception ontology (35 labels) demonstrates high top-$1$ and top-$k$ accuracy on a reproducible protocol, along with low violation rates and strong rerun stability on an instrumented subset. The paper emphasizes system engineering choices for reproducibility and failure containment rather than model training.

**Index Terms**—misconception diagnosis, educational data mining, retrieval-constrained LLMs, RAG, LLM-as-a-judge, reliability, short-answer assessment.

## 1. Introduction
Educational interventions are most effective when they address the *reasons* a learner is wrong rather than only the correctness of an answer. A misconception is a semantic and often internally consistent mental model that can generate recurring errors across contexts. In contrast, grading (or scoring) is typically a surface-level decision about correctness (or partial credit) with limited obligations to explain *why* an error occurred.

This distinction matters operationally: an automated system that “grades” can be wrong in a bounded way (e.g., one point off), whereas a system that *diagnoses misconceptions* can cause pedagogical harm if it confidently assigns an ungrounded misconception label or invents a plausible-sounding explanation. Such hallucinations are especially problematic in educational contexts, where outputs may inform instructional decisions and student feedback loops.

Recent work has explored NLP and LLM-based approaches for misconception identification and short-answer scoring across domains (e.g., physics, environmental science, programming, and mathematics) [2], [3], [4], [5]. However, unconstrained LLM “grader” or “diagnoser” pipelines often lack explicit mechanisms for bounding outputs to a closed misconception ontology and for ensuring that explanations are grounded in evidence. Further, many approaches emphasize model accuracy but under-specify *system reliability*, including schema compliance, stability under reruns, and operational failure modes.

This paper treats misconception diagnosis as a *systems engineering* problem: how to design an LLM-enabled pipeline that is safer and more reproducible by construction, without fine-tuning the underlying LLM.

### Contributions
- **Retrieval-constrained diagnosis architecture** that constructs a bounded hypothesis space from retrieved evidence and constrains the LLM to rank/select among candidates only.
- **Reliability layer via LLM-as-a-judge** that audits schema compliance, candidate-set adherence, evidence grounding, and rerun stability.
- **Closed-world evaluation on MAP** using a fixed misconception ontology and reproducible protocol, including accuracy, violation rate, and stability metrics.

## 2. Related Work
### 2.1 Misconception Detection and Taxonomy-Based Diagnosis
Prior misconception detection research frequently adopts supervised classification over expert-coded labels. For example, Demirezen *et al.* propose transformer-based and classical baselines for detecting physics misconceptions from open-ended responses, emphasizing the costs and subjectivity of manual coding and the appeal of automated pipelines [2]. Kökver *et al.* study NLP and supervised learning to identify misconceptions in an environmental science context (greenhouse effect), also highlighting expert agreement as a practical evaluation concern [3].

These works motivate two system requirements for mathematics misconception diagnosis: (i) a *fixed ontology* (closed-world labels), and (ii) *evidence-based justification* to support human review and instructional accountability.

### 2.2 Misconception Benchmarks and Math-Specific Diagnosis
Otero *et al.* propose a benchmark for middle-school algebra misconceptions and evaluate LLM-based diagnosis under controlled conditions, reporting the value of topic restriction and human expertise in improving outcomes [1]. This supports the broader point that constraint mechanisms (e.g., topic restriction or candidate restriction) can improve reliability and interpretability in misconception diagnosis systems.

### 2.3 Automated Short-Answer Assessment and LLM Graders
Short-answer scoring is related but not equivalent to misconception diagnosis. Chamieh *et al.* evaluate LLMs for short answer scoring across datasets and highlight limitations of zero-/few-shot LLM scoring, with fine-tuning often needed for competitive performance [6]. Speiser and Weng examine LLM-based grading on a programming short-answer dataset and discuss error types and practical concerns [7]. Grévisse studies LLM-based grading in medical education and explicitly discusses consistency, bias, and hallucination risks in LLM-based assessment pipelines [5]. Abeywardana *et al.* explore prompt engineering and retrieval augmentation to improve automated grading and transparency [8].

These studies motivate separating *diagnosis* from *scoring*, and designing constraints and reliability checks appropriate for diagnosis.

### 2.4 LLM-as-a-Judge and Reliability Engineering
LLM-as-a-judge has been proposed as a scalable evaluation mechanism for LLM outputs. Boyapati *et al.* present an adaptive pipeline (LevelEval) for evaluating LLM judges across tasks [9]. Rafique and Marsden discuss cloud-native evaluation workflows using LLM-as-a-judge [10]. Such work suggests that an LLM judge can be treated as a system component that enforces structured checks (schema compliance, consistency, and policy constraints), rather than as an oracle.

### 2.5 Retrieval-Augmented Generation and Hallucination Mitigation
Retrieval augmentation is widely used to reduce hallucinations and increase grounding by conditioning generation on external evidence. In educational and assessment settings, retrieval can provide traceable support for decisions and enable bounded reasoning. Qiu *et al.* propose SteLLA, a structured short-answer grading system that uses retrieval augmentation over instructor-provided reference answers and rubrics to ground evaluation and feedback [11].

## 3. Problem Definition
Let $\mathcal{L}$ be a fixed misconception ontology (closed-world label set). Each item is a tuple $(q, a, s)$ where $q$ is the question prompt (including any image description), $a$ is the reference answer (used only as context), and $s$ is the student’s free-text explanation. The goal is to predict a misconception label $\ell \in \mathcal{L}$ for misconception-bearing responses.

This paper focuses on *diagnosis*, not grading. In particular:
- **Misconception diagnosis**: infer $\ell$ that best captures the student’s underlying misunderstanding.
- **Non-goal**: generating novel misconceptions, open-world discovery, or assigning partial credit.

We cast diagnosis as a constrained reasoning task:
1. Retrieve evidence $E$ from an indexed memory of expert-labeled examples.
2. Construct candidate labels $C(E) \subseteq \mathcal{L}$ *only* from retrieved evidence.
3. Constrain an LLM to output a selection/ranking over $C(E)$ under a fixed schema.

## 4. System Design
The system is engineered as a retrieval-first pipeline that explicitly bounds the hypothesis space and contains failure modes.

### 4.1 Retrieval Component (Evidence-First)
**Indexed memory.** The system maintains an index over labeled training exemplars, each storing the concatenated text: question, reference answer (context), and student explanation. Each exemplar is labeled with a misconception label $\ell$.

**Hybrid retrieval.** We implement hybrid retrieval using dense embeddings (Sentence Transformers) and lexical BM25 (Tantivy), fused via reciprocal rank fusion (RRF). This mirrors common hybrid retrieval designs for robustness to vocabulary mismatch.

### 4.2 Candidate Construction (Bounded Hypothesis Space)
Given a test instance, retrieval returns top-$k$ evidence exemplars $E = \{e_1, \ldots, e_k\}$. Candidate labels are derived as:
$$
C(E) = \mathrm{Unique}\left(\{\ell(e_i)\}_{i=1}^k\right)
$$
and optionally capped to a maximum size (to prevent prompt explosion).

This step is a deliberate safety constraint: the LLM cannot introduce labels outside $C(E)$, and $C(E)$ is grounded in retrieved evidence.

### 4.3 Diagnoser LLM (Constrained Ranking/Selection)
The diagnoser LLM receives:
- The input $(q,a,s)$,
- The candidate label list $C(E)$,
- A small set of retrieved exemplars $E$ (student text snippets and their labels).

The LLM is instructed to output **strict JSON**:
- `ranked_labels`: an ordered list from $C(E)$,
- `selected_label`: the top prediction,
- `evidence_ids`: identifiers of retrieved exemplars supporting the decision.

**No fine-tuning** is performed; the system relies on prompt constraints and retrieval-bounded candidates.

### 4.4 Judge LLM (Reliability Component)
A secondary LLM is used as a *reliability component* that audits:
- **Schema compliance**: required JSON keys and types,
- **Candidate adherence**: selected label is in $C(E)$,
- **Evidence linkage**: evidence IDs refer to retrieved exemplars,
- **Grounding**: decision is plausibly supported by retrieved evidence,
- **Stability**: repeated diagnoser runs should agree at a high rate.

The judge is not treated as ground truth. Instead, it provides a structured signal for system monitoring and failure containment, aligned with LLM-judge pipelines studied in prior work [9], [10].

## 5. Experimental Setup
### 5.1 Dataset: MAP (Closed-World Benchmark)
We evaluate on a closed-world MAP benchmark release provided as [map-data.csv](map-data.csv). The benchmark properties relevant to this paper are:
- **Middle-school mathematics** short-answer explanations.
- **Fixed misconception ontology** with 35 misconception labels.
- **Expert-annotated misconception labels** for misconception-bearing responses.

In the provided release, there are 15 unique questions and 9,860 misconception-labeled student explanations (categories `True_Misconception` and `False_Misconception`). The label distribution is imbalanced (majority-class accuracy $\approx 14.7\%$ over misconception-labeled rows).

This paper does **not** address open-world discovery; all predictions are restricted to the closed label set.

### 5.2 Protocol and Splits
We use an 80/20 split *within each question ID* (to avoid trivially selecting near-duplicate training examples from a different question while still reflecting item-specific deployment). The split is deterministic under a fixed random seed.

### 5.3 Baselines
We report simple, reproducible baselines that do not require training:
- **Majority class** on training labels.
- **Embedding nearest neighbor**: predict the label of the nearest training exemplar by cosine similarity.
- **Hybrid retrieval vote**: retrieve evidence and choose the most frequent label among retrieved exemplars.

These baselines are included to contextualize the system, not to claim state-of-the-art.

### 5.4 Implementation and Reproducibility
All evaluation code is provided in [eval_map_system.py](eval_map_system.py). The diagnoser and judge run locally via Ollama:
- Diagnoser: `llama3.2:latest`
- Judge: `gemma3:1b`

The retriever uses Sentence Transformers embeddings (`all-MiniLM-L6-v2`) and Tantivy BM25 with RRF fusion. No fine-tuning is performed.

## 6. Results
This section reports only metrics computed from the provided dataset and scripts.

### 6.1 Accuracy and Constraint Violations
On a reproducible evaluation run with 80 test examples (seeded split; retrieval candidates capped; strict JSON enforced), the retrieval-constrained LLM system achieved:
- **Top-1 accuracy:** 92.50% (74/80)
- **Top-3 accuracy:** 98.75% (79/80)
- **Violation rate:** 1.25% (1/80), defined as schema failure or selecting a label outside the retrieved candidate set

Baselines on the same run:
- Majority class: 12.50% (10/80)
- Embedding nearest neighbor: 96.25% (77/80)
- Hybrid retrieval vote: 93.75% (75/80)

The strong nearest-neighbor baseline suggests that for this closed-world benchmark release, local semantic similarity to labeled exemplars is often sufficient. The retrieval-constrained LLM’s primary added value is not raw accuracy but *structured outputs* (ranked candidates + evidence IDs) and *constraint satisfaction* suitable for audited educational workflows.

### 6.2 Stability Across Reruns (Instrumented Subset)
On a fully instrumented run of 30 test examples (three independent diagnoser runs per example using different seeds; judge enabled), the system achieved:
- **Top-1 accuracy:** 93.33% (28/30)
- **Top-3 accuracy:** 100.00% (30/30)
- **Violation rate:** 0.00% (0/30)
- **3-run exact agreement:** 93.33% (28/30)

This stability metric operationalizes the requirement that educational outputs be repeatable; instability can be surfaced as a reliability issue even when accuracy is high.

### 6.3 Judge Audits
On the same 30-example instrumented run, the judge reported:
- **Schema + candidate adherence OK:** 100% (30/30)
- **Grounded in retrieved evidence:** 100% (30/30)

These judge signals are treated as monitoring indicators rather than ground truth; they provide a structured audit trail and can trigger fallbacks (e.g., “abstain” or human review) in deployments.

## 7. Discussion
### 7.1 Strengths
- **Bounded outputs reduce risk.** Candidate construction from retrieved evidence prevents open-ended label invention and makes the hypothesis space explicit.
- **Auditable decisions.** Evidence IDs and judge audits provide artifacts for human review and for debugging retrieval failures.
- **Reproducible, portable evaluation.** The system uses off-the-shelf components (embeddings, BM25, local LLM runtime) and a scripted protocol.

### 7.2 Limitations
- **Dependence on retrieval quality.** If retrieval fails to surface relevant exemplars, the candidate set can exclude the true label; the LLM cannot recover by design.
- **Within-question splits may overestimate generalization.** The current protocol reflects item-level deployment but does not test cross-item transfer; stronger protocols could hold out entire questions.
- **Judge is not ground truth.** LLM-as-a-judge can be biased and should be calibrated against human audits when used in high-stakes settings.

### 7.3 Implications for Educational AI Systems
In educational contexts, a “safe” misconception diagnosis system should prioritize:
- Transparent constraints (closed-world labels, bounded candidates),
- Evidence-grounded decisions,
- Reliability monitoring (schema, adherence, stability),
over unconstrained generation.

This positioning explicitly contrasts with (i) pure supervised classifiers trained end-to-end (which can be accurate but opaque and brittle under shift) [2], [3], (ii) unconstrained LLM graders/diagnosers that may invent labels or rationales [5], and (iii) fine-tuned task-specific LLMs that require ongoing training data and introduce portability concerns [6].

## 8. Threats to Validity
- **Dataset provenance and scope.** The evaluation uses the provided MAP release [12]; conclusions may not generalize to other MAP splits, broader grade levels, or richer ontologies.
- **Sampling variance.** Reported metrics include an 80-example run and a 30-example instrumented run; larger evaluations may shift estimates.
- **Prompt and model dependence.** Results depend on the chosen local LLM and prompt templates; different models may exhibit different violation/stability behavior.
- **Potential near-duplicate leakage.** Short explanations may contain repeated patterns; nearest-neighbor baselines may benefit from templatic similarity.

## 9. Conclusion and Future Work
This paper presented a retrieval-constrained LLM system for closed-world misconception diagnosis in mathematics. The key system principle is to treat retrieval as a safety mechanism that constructs a bounded hypothesis space, and to treat a judge model as a reliability component that audits outputs for adherence, grounding, and stability. Evaluation on a MAP benchmark release demonstrates strong top-$k$ accuracy with low violation rates and high rerun agreement on an instrumented subset.

Future work should extend the system with (i) **human-in-the-loop (HITL)** review workflows and rubric-driven adjudication policies, and (ii) careful extensions to **open-world** settings where new misconception variants can be proposed but must be vetted and incorporated into the ontology. These extensions are explicitly out of scope for the present paper.

## References
[1] N. Otero, S. Druga, and A. Lan, “A Benchmark for math misconceptions: bridging gaps in middle school algebra with AI-supported instruction,” 2025.

[2] M. U. Demirezen, O. Yilmaz, and E. Ince, “New models developed for detection of misconceptions in physics with artificial intelligence,” 2023.

[3] Y. Kökver, H. M. Pektaş, and H. Çelik, “Artificial intelligence applications in education: Natural language processing in detecting misconceptions,” 2024.

[4] B. Fischer, F. Birk, E.-M. Iwer, S. E. Panitz, and R. Dorner, “Addressing Misconceptions in Introductory Programming: Automated Feedback in Integrated Development Environments,” in *Proceedings of the 15th International Conference on Education Technology and Computers (ICETC)*, 2023.

[5] C. Grévisse, “LLM-based automatic short answer grading in undergraduate medical education,” 2024.

[6] I. Chamieh, T. Zesch, and K. Giebermann, “LLMs in Short Answer Scoring: Limitations and Promise of Zero-Shot and Few-Shot Approaches,” 2024.

[7] S. Speiser and A. Weng, “Enhancing Short Answer Grading With OpenAI APIs,” 2024.

[8] T. Abeywardana, N. Nandadewa, and V. Wickramasinghe, “Enhancing Automated Grading with Capabilities of LLMs: Using Prompt Engineering and RAG Techniques,” 2025.

[9] M. Boyapati, L. Meesala, R. Aygun, and B. Franks, “LevelEval: Adaptive Pipeline for Evaluating LLM as a Judge - Analysis on Open LLMs as Judges,” 2024.

[10] A. Rafique and B. D. Marsden, “Automated LLM Deployment and Evaluation: A Cloud-Native Approach Using LLM-as-a-Judge,” 2025.

[11] H. Qiu, B. White, A. Ding, R. Costa, A. Hachem, W. Ding, and P. Chen, “SteLLA: A Structured Grading System Using LLMs with RAG,” 2024.

[12] MAP (Charting Student Math Misunderstandings) benchmark release, [map-data.csv](map-data.csv), accessed 2025-12-28.
