import nltk
from nltk.tokenize import sent_tokenize

def _ensure_punkt_downloaded():
    """Ensures sentence boundary models are present in the local cache."""
    try:
        nltk.data.find('tokenizers/punkt')
    except (LookupError, OSError):  # <--- Added OSError here
        nltk.download('punkt', quiet=True)
        
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except (LookupError, OSError):  # <--- Added OSError here
        nltk.download('punkt_tab', quiet=True)

class TextProcessor:
    def __init__(self):
        _ensure_punkt_downloaded()

    @staticmethod
    def split_into_sentences(text: str) -> list[str]:
        """
        Splits raw text into individual atomic sentence claims.
        Strips whitespace and removes empty strings.
        """
        raw_sentences = sent_tokenize(text.strip())
        return [s.strip() for s in raw_sentences if s.strip()]

# --- Standalone Test Execution ---
if __name__ == "__main__":
    processor = TextProcessor()
    
    sample_text = (
        "Database node-01 crashed at 02:14 UTC. The outage was traced to an OOM killer event. "
        "Dr. Smith verified the heap allocation exceeded 64GB."
    )
    
    sentences = processor.split_into_sentences(sample_text)
    print("\n--- Text Processor Test ---")
    print(f"Extracted {len(sentences)} atomic sentences:")
    for idx, sentence in enumerate(sentences, 1):
        print(f"  [{idx}] {sentence}")
