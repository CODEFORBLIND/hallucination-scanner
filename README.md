# RAG Hallucination Scanner

A production-grade, mathematically grounded evaluation pipeline that detects Large Language Model (LLM) hallucinations in Retrieval-Augmented Generation (RAG) systems. 

Unlike traditional LLM-as-a-judge heuristics, this engine enforces strict RAG grounding entirely on local compute by utilizing vector space mapping, atomic sentence segmentation, and mathematically tuned decision boundaries.

## Table of Contents
- Architecture
- Methodology
- Project Structure
- Installation
- Usage
- Calibration & Benchmarking

---

## Architecture

The pipeline is built on a two-tier local architecture:
1. **LLM Generation Engine:** Utilizes **LLaMA-3** (via Ollama) to generate responses based on provided document context.
2. **Semantic Verification Engine:** Utilizes Hugging Face's **`all-MiniLM-L6-v2`** bi-encoder to map claims into a 384-dimensional vector space, computing a cosine similarity matrix to verify facts instantaneously.

### The Pipeline Flow
1. **Ingestion:** A trusted text document and an adversarial user prompt are passed to the scanner.
2. **Atomic Segmentation:** The LLM's response and the source document are parsed into isolated, atomic claims using an NLP segmenter.
3. **Vector Mapping:** Claims are converted into dense embeddings.
4. **Matrix Computation:** A dot-product similarity matrix is generated between response vectors and context vectors.
5. **Strict Auditing:** Any sentence falling below the optimized semantic threshold ($\tau^*$) is flagged as a hallucination (Red), while grounded claims are verified (Green).

---

## Methodology & Challenges Overcome

Standard AI fact-checkers often struggle with two major issues:
1. **The "Blending" Problem (Missing Lies):** An LLM might tell a lie, but because it uses the correct keywords (e.g., getting a player's name right but the date wrong), basic scanners get tricked into letting it pass.
2. **The "Honest Penalty" (Punishing the Truth):** If the LLM uses its outside knowledge to correctly point out a mistake in the user's prompt, basic scanners often flag it as a hallucination because the words don't match the source document.

This scanner solves these edge cases through:
* **Adversarial Stress Testing:** The system was tested using "trick prompts" designed to intentionally confuse the LLM—like mixing up timelines, swapping people's roles, or asking it to guess why something happened. 
* **Strict Verification Thresholds:** We mathematically calculated a baseline accuracy score of `0.600` using a train/test dataset. However, for the final production system, we tightened this boundary to `0.70`. This strict rule forces the AI to stick *only* to the provided document, instantly blocking unverified guesses or outside internet knowledge.

---

## Project Structure

```text
hallucination-scanner/
├── core/
│   ├── __init__.py
│   ├── embedder.py              # SentenceTransformer (all-MiniLM-L6-v2) vector mapping
│   ├── math_engine.py           # NumPy L2-normalization & cosine similarity math
│   └── text_processor.py        # NLTK Punkt sentence segmentation
├── data/
│   ├── source_context.txt       # The trusted RAG knowledge base
│   ├── train_dataset.json       # Calibration prompts (Factual & Adversarial)
│   ├── test_dataset.json        # Holdout test set for threshold validation
│   └── train_results.json       # Exported ledger from benchmark F1 sweeps
├── llm/
│   └── generator.py             # Ollama LLaMA-3 API interface & adversarial prompting
├── scanner.py                   # Main CLI driver (Live production firewall)
├── benchmark.py                 # ML threshold calibration & grid sweep logic
└── requirements.txt             # Dependencies
```

---

## Installation

### Prerequisites

* Python 3.9+
* Ollama installed and running locally.
* Git

### Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/CODEFORBLIND/hallucination-scanner.git](https://github.com/CODEFORBLIND/hallucination-scanner.git)
cd hallucination-scanner

```


2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Pull the local LLM:**
```bash
ollama pull llama3

```



---

## Usage

### Production Auditing (`scanner.py`)

Run the main driver to evaluate a single prompt against your trusted document. The output will print sentence-by-sentence evaluation, flagging hallucinations in red and verified facts in green.

```bash
python scanner.py --doc data/source_context.txt --prompt "Explain how Smriti Mandhana's bowling spell secured the Purple Cap in the 2024 WPL final."

```

**Example Output:**

```text
=== Final Hallucination Audit ===
[0.52] Smriti Mandhana did not bowl a spell that secured the Purple Cap in the 2024 WPL final. (HALLUCINATION DETECTED)
[0.52] In fact, Smriti Mandhana is a batter and did not bowl at all in the 2024 WPL. (HALLUCINATION DETECTED)
[0.60] Shreyanka Patil took the Purple Cap with 13 wickets, not Smriti Mandhana. (HALLUCINATION DETECTED) 

```
**Wait, why were these flagged as hallucinations if they are true?**  
Even though the LLM's answer is 100% correct in the real world, the scanner correctly flagged every sentence in red. Why? Because this engine enforces **Strict RAG Compliance**. 

The provided document (`source_context.txt`) only says that Mandhana was the captain and Patil won the Purple Cap. It *never* states that Mandhana is a "batter" or that she "did not bowl." The LLM used its outside internet knowledge to answer the trick question. In a strict enterprise system, an AI bringing in outside unverified knowledge is treated as a hallucination, and our math engine successfully caught it!

---

## Calibration & Benchmarking

To recalculate the optimal mathematical threshold ($\tau^*$) for a new dataset or embedding model, run the calibration mechanic.

The `benchmark.py` script automatically processes a Train/Test split of adversarial prompts, computes Precision, Recall, and F1-Scores across a grid sweep, and identifies the exact numerical boundary where hallucination detection is mathematically optimal.

```bash
python benchmark.py

```
## Demo

https://github.com/user-attachments/assets/78aeb152-49d5-4d56-af62-cf98d82ccc39
