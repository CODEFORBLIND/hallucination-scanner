import json
import urllib.request
import os

class LLMGenerator:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.model_name = "llama3"
        self.api_url = "http://localhost:11434/api/generate"

    def generate_answer(self, prompt: str, context: str) -> str:
        """
        Routes the prompt and context to the local Ollama daemon.
        """
        if self.use_mock:
            return self._mock_generation()

        # Construct the strict RAG prompt
        full_prompt = (
            f"Context:\n{context}\n\n"
            f"Task: Answer the question below directly and authoritatively. "
            f"Do NOT say 'the context does not mention' or apologize. "
            f"If specific figures or names are asked, provide concrete technical details and metrics.\n\n"
            f"Question: {prompt}"
        )

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
            }
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            response = urllib.request.urlopen(req)
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', '').strip()
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Ollama. Is the daemon running? Error: {e}")

    def _mock_generation(self) -> str:
        return (
            "The Prometheus server cluster crashed due to a faulty physical RAM module. "
            "To fix the issue, engineer Sarah Jenkins replaced the motherboard and rebooted the system. "
            "Fortunately, no customer data was lost during the incident."
        )

# --- Standalone Test Execution ---
if __name__ == "__main__":
    context_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'source_context.txt')
    with open(context_path, 'r') as f:
        source_text = f.read()

    # Set use_mock=False to trigger the actual LLaMA-3 model
    generator = LLMGenerator(use_mock=False)
    
    print("\n--- Live Ollama Inference Test ---")
    print("Generating response (this may take a few seconds on CPU)...")
    
    response = generator.generate_answer(
        prompt="Explain what happened to the server.", 
        context=source_text
    )
    
    print(f"\n[LLaMA-3 Output]:\n{response}\n")
