import os
import argparse
from core.embedder import VectorEngine
from core.text_processor import TextProcessor
from core.math_engine import SimilarityEngine
from llm.generator import LLMGenerator
from utils.terminal_ui import TerminalUI

# The geometric threshold for cosine similarity. 
# Scores below 0.50 indicate the vector exists outside the context's semantic space.
THRESHOLD = 0.70 

def main():
    parser = argparse.ArgumentParser(description="RAG Hallucination Scanner")
    parser.add_argument("--doc", type=str, required=True, help="Path to the source text file")
    parser.add_argument("--prompt", type=str, required=True, help="Question to ask the LLM")
    args = parser.parse_args()
    TerminalUI.print_header("RAG Hallucination Scanner Initialization")
    
    # 1. Boot up hardware and load weights into RAM
    embedder = VectorEngine()
    text_processor = TextProcessor()
    llm = LLMGenerator(use_mock=False)
    math_engine = SimilarityEngine()
    
    # 2. Load Ground Truth Context
    context_path = os.path.join(os.path.dirname(__file__), 'data', 'source_context.txt')
    with open(context_path, 'r') as f:
        source_context = f.read()
        
    # 3. Generate LLM Answer
    TerminalUI.print_header(f"Asking LLM: '{args.prompt}'")
    
    # We pass args.prompt directly from the terminal command
    llm_response = llm.generate_answer(args.prompt, source_context) 
    
    print(f"Raw Output:\n{llm_response}\n")
    # 4. Process & Split Text (NLP Segmentation)
    TerminalUI.print_header("Running NLP Sentence Segmentation")
    context_sentences = text_processor.split_into_sentences(source_context)
    generated_sentences = text_processor.split_into_sentences(llm_response)
    print(f"Segmented context into {len(context_sentences)} atomic claims.")
    print(f"Segmented response into {len(generated_sentences)} atomic claims.")
    
    # 5. Embed to Vector Space (Forward Pass)
    TerminalUI.print_header("Executing Vector Space Mapping")
    C_matrix = embedder.embed_sentences(context_sentences)
    S_matrix = embedder.embed_sentences(generated_sentences)
    print(f"Context Matrix Shape: {C_matrix.shape}")
    print(f"Response Matrix Shape: {S_matrix.shape}")
    
    # 6. Run Matrix Calculus for Similarity
    TerminalUI.print_header("Computing Cosine Similarity Matrix")
    support_scores = math_engine.compute_max_support(S_matrix, C_matrix)
    
    # 7. Render Final Audit
    TerminalUI.print_header("Final Hallucination Audit")
    for sentence, score in zip(generated_sentences, support_scores):
        TerminalUI.print_result(sentence, score, THRESHOLD)
    print("\n")

if __name__ == "__main__":
    main()
