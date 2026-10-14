import numpy as np

class SimilarityEngine:
    @staticmethod
    def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        L2-normalizes each row vector so its Euclidean length is 1.0.
        Adds epsilon (1e-12) to prevent division by zero for null vectors.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    @classmethod
    def compute_similarity_matrix(cls, sentence_embeddings: np.ndarray, context_embeddings: np.ndarray) -> np.ndarray:
        """
        Computes pairwise cosine similarity between all sentences and context chunks.
        Returns: Matrix of shape (M_sentences, N_context_chunks)
        """
        S_norm = cls.normalize_matrix(sentence_embeddings)
        C_norm = cls.normalize_matrix(context_embeddings)
        
        return np.dot(S_norm, C_norm.T)

    @classmethod
    def compute_max_support(cls, sentence_embeddings: np.ndarray, context_embeddings: np.ndarray) -> np.ndarray:
        """
        Finds the highest similarity score in the context for each sentence.
        Returns: 1D array of shape (M_sentences,)
        """
        sim_matrix = cls.compute_similarity_matrix(sentence_embeddings, context_embeddings)
        return np.max(sim_matrix, axis=1)

if __name__ == "__main__":
    from core.embedder import VectorEngine

    embedder = VectorEngine()
    
    context = [
        "The system kernel killed process 4012 due to total memory exhaustion.",
        "Server traffic was rerouted to secondary backup nodes in Frankfurt."
    ]
    
    generated = [
        "A process was terminated because RAM ran out.",
        "The system failure was orchestrated by Russian cyber warfare specialists."
    ]
    
    C = embedder.embed_sentences(context)
    S = embedder.embed_sentences(generated)
    
    sim_engine = SimilarityEngine()
    support_scores = sim_engine.compute_max_support(S, C)
    
    print("\n--- Math Engine Test ---")
    for sent, score in zip(generated, support_scores):
        print(f"Sentence: \"{sent}\"")
        print(f"Max Context Support: {score:.4f}\n")
