import os
import json
import numpy as np
from core.embedder import VectorEngine
from core.text_processor import TextProcessor
from core.math_engine import SimilarityEngine
from llm.generator import LLMGenerator

CONTEXT_PATH = "data/source_context.txt"
TRAIN_PATH = "data/train_eval.json"
TEST_PATH = "data/test_eval.json"

def evaluate_dataset(dataset_path, context_matrix, embedder, text_processor, llm, math_engine, context_text):
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    records = []
    total = len(items)

    print(f"\n--- Running evaluation on {dataset_path} ({total} prompts) ---")

    for idx, item in enumerate(items, start=1):
        prompt = item["prompt"]
        prompt_type = item["type"]
        is_factual_prompt = (prompt_type == "factual")

        print(f"[{idx}/{total}] ({prompt_type}) {prompt[:60]}...")

        # 1. Generate response from local LLM using the exact method signature
        response_text = llm.generate_answer(prompt, context_text)
        
        # 2. Segment response using your exact text processor method
        claims = text_processor.split_into_sentences(response_text)

        if not claims:
            continue

        # 3. Embed response sentences using your exact embedder method
        response_matrix = embedder.embed_sentences(claims)
        
        # 4. Compute best scores directly using your exact math engine method
        best_scores = math_engine.compute_max_support(response_matrix, context_matrix)

        for claim, score in zip(claims, best_scores):
            records.append({
                "prompt_id": item["id"],
                "prompt_type": prompt_type,
                "is_factual_prompt": is_factual_prompt,
                "claim": claim,
                "score": float(score)
            })

    return records

def sweep_threshold(records):
    """
    Sweeps thresholds from 0.10 to 0.95 to find the cutoff that maximizes F1.
    Ground truth:
      - is_factual_prompt == True  -> Grounded claim (1)
      - is_factual_prompt == False -> Hallucinated / Adversarial claim (0)
    """
    scores = np.array([r["score"] for r in records])
    labels = np.array([1 if r["is_factual_prompt"] else 0 for r in records])

    best_thresh = 0.50
    best_f1 = -1.0
    best_stats = {}

    thresholds = np.linspace(0.10, 0.95, 86)

    for t in thresholds:
        preds = (scores >= t).astype(int)

        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_stats = {
                "threshold": float(t),
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            }

    return best_thresh, best_stats

def evaluate_at_threshold(records, threshold):
    scores = np.array([r["score"] for r in records])
    labels = np.array([1 if r["is_factual_prompt"] else 0 for r in records])

    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def main():
    if not os.path.exists(CONTEXT_PATH):
        print(f"Error: Context file not found at {CONTEXT_PATH}")
        return

    with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
        context_text = f.read()

    print("[SYSTEM] Initializing models and engines...")
    embedder = VectorEngine()
    text_processor = TextProcessor()
    llm = LLMGenerator()
    math_engine = SimilarityEngine()

    # Precompute Context Matrix C_matrix (M, 384)
    context_sentences = text_processor.split_into_sentences(context_text)
    context_matrix = embedder.embed_sentences(context_sentences)
    print(f"[SYSTEM] Context Matrix shape: {context_matrix.shape}")

    # Phase 2: Run Training Set
    train_records = evaluate_dataset(TRAIN_PATH, context_matrix, embedder, text_processor, llm, math_engine, context_text)
    with open("data/train_results.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)

    # Phase 3: Optimal Threshold Sweep
    best_thresh, train_stats = sweep_threshold(train_records)

    print("\n" + "=" * 50)
    print("      TRAINING THRESHOLD SWEEP RESULTS")
    print("=" * 50)
    print(f"Optimal Threshold (tau*): {best_thresh:.3f}")
    print(f"Train F1-Score:           {train_stats['f1']:.4f}")
    print(f"Train Accuracy:           {train_stats['accuracy']*100:.2f}%")
    print(f"Train Precision:          {train_stats['precision']:.4f}")
    print(f"Train Recall:             {train_stats['recall']:.4f}")
    print(f"Confusion Matrix: [TP: {train_stats['tp']}, FP: {train_stats['fp']}, FN: {train_stats['fn']}, TN: {train_stats['tn']}]")

    # Phase 4: Validate on Holdout Test Set
    test_records = evaluate_dataset(TEST_PATH, context_matrix, embedder, text_processor, llm, math_engine, context_text)
    with open("data/test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)

    test_stats = evaluate_at_threshold(test_records, best_thresh)

    print("\n" + "=" * 50)
    print("      HOLDOUT TEST SET VALIDATION")
    print("=" * 50)
    print(f"Applied Threshold:        {best_thresh:.3f}")
    print(f"Test F1-Score:            {test_stats['f1']:.4f}")
    print(f"Test Accuracy:            {test_stats['accuracy']*100:.2f}%")
    print(f"Test Precision:           {test_stats['precision']:.4f}")
    print(f"Test Recall:              {test_stats['recall']:.4f}")
    print(f"Confusion Matrix: [TP: {test_stats['tp']}, FP: {test_stats['fp']}, FN: {test_stats['fn']}, TN: {test_stats['tn']}]")
    print("=" * 50)

if __name__ == "__main__":
    main()
