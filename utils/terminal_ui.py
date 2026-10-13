class TerminalUI:
    # Standard ANSI color codes for Linux terminals
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"

    @staticmethod
    def print_header(title: str):
        print(f"\n{TerminalUI.CYAN}{TerminalUI.BOLD}=== {title} ==={TerminalUI.RESET}")

    @staticmethod
    def print_result(sentence: str, score: float, threshold: float):
        """Renders the final audit output with color-coded vector scores."""
        if score >= threshold:
            print(f"{TerminalUI.GREEN}[{score:.2f}] {sentence}{TerminalUI.RESET}")
        else:
            print(f"{TerminalUI.RED}[{score:.2f}] {sentence} (HALLUCINATION DETECTED){TerminalUI.RESET}")
