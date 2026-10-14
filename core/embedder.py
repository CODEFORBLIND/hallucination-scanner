import numpy as np
from sentence_transformers import SentenceTransformer

class VectorEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the embedding model. 
        Downloads the 90MB weights to your local cache on the first run.
        """
        print(f"[SYSTEM] Booting Vector Engine: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("[SYSTEM] Vector Engine Online.")

    def embed_sentences(self, sentences: list[str]) -> np.ndarray:
        """
        Takes a list of strings and returns a 2D NumPy array (Matrix).
        Shape: (Number of Sentences, 384)
        """
        embeddings = self.model.encode(sentences)
        return np.array(embeddings)

if __name__ == "__main__":
    engine = VectorEngine()
    
    test_data = [
        "The server crashed at 2 AM due to a memory leak.",
        "The admin was asleep."
    ]
    
    matrix = engine.embed_sentences(test_data)
    
    print("\n--- Hardware Test ---")
    print(f"Input Sentences: {len(test_data)}")
    print(f"Matrix Shape: {matrix.shape} (Sentences x Dimensions)")
    print(f"Vector Data [Row 0, first 5 floats]: \n{matrix[0][:5]}")
