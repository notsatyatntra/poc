from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import re
from typing import List, Tuple, Dict, Optional
from backend_langgraph.llm_class import LLMManager, LLMConfig
# from llm_class import LLMManager, LLMConfig

class UtilConfig:
    """Configuration class for utility functions."""
    
    def __init__(
        self,
        web_search_examples: Optional[List[Dict[str, str]]] = None,
        context_examples: Optional[List[Dict[str, str]]] = None,
        url_pattern_with_protocol: str = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        url_pattern_no_protocol: str = r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[a-zA-Z0-9-._~:/?#[\]@!$&\'()*+,;=]*)?',
        llm_config: Optional[LLMConfig] = None
    ):
        """
        Args:
            web_search_examples (List[Dict[str, str]]): Few-shot examples for web search classification.
            context_examples (List[Dict[str, str]]): Few-shot examples for context classification.
            url_pattern_with_protocol (str): Regex for URLs with protocol (http/https).
            url_pattern_no_protocol (str): Regex for domain names without protocol.
            llm_config (LLMConfig): Configuration for the LLMManager.
        """
        # Default few-shot examples for web search
        self.web_search_examples = web_search_examples or [
            {"question": "Who won the current/latest FIFA World Cup?", "label": "needs web search"},
            {"question": "What is the latest stock price of Apple?", "label": "needs web search"},
            {"question": "What is the capital of France?", "label": "answered directly using LLM"},
            {"question": "Explain the theory of relativity?", "label": "answered directly using LLM"},
        ]
        
        # Default few-shot examples for context
        self.context_examples = context_examples or [
            {"question": "Who won the current/latest FIFA World Cup?", "label": "needs context"},
            {"question": "What is the latest stock price of Apple?", "label": "needs context"},
            {"question": "What is the capital of France?", "label": "answered directly using LLM"},
            {"question": "Explain the theory of relativity?", "label": "answered directly using LLM"},
        ]
        
        self.url_pattern_with_protocol = url_pattern_with_protocol
        self.url_pattern_no_protocol = url_pattern_no_protocol
        self.llm_config = llm_config or LLMConfig()

class UtilManager:
    """Manages utility functions for determining web search needs, context, and URL handling."""
    
    def __init__(self, config: UtilConfig):
        """
        Initialize the UtilManager with a configuration.

        Args:
            config (UtilConfig): Configuration object for utility settings.
        """
        self.config = config
        self.llm_manager = LLMManager(self.config.llm_config)  # Initialize LLMManager

    def _build_few_shot_prompt(self, prompt: str, examples: List[Dict[str, str]], label_key: str) -> str:
        """Build a few-shot prompt for LLM classification."""
        few_shot_prompt = f"""
        You are an intelligent system that determines whether a query requires a {label_key}.
        Based on the given examples, classify the new question accordingly.
        Strictly give only the labels as the response specified in the examples above.

        Examples:
        """
        for example in examples:
            few_shot_prompt += f"\nQuestion: {example['question']}\nLabel: {example['label']}\n"
        
        few_shot_prompt += f"\nQuestion: {prompt}\nLabel: "
        return few_shot_prompt

    def few_shot_needs_web_search(self, prompt: str) -> bool:
        """Determine if a web search is needed using few-shot prompting with LLM."""
        few_shot_prompt = self._build_few_shot_prompt(prompt, self.config.web_search_examples, "web search")
        response = self.llm_manager.call_llm(few_shot_prompt)
        # print(response)
        return "needs web" in response.lower()

    def needs_context(self, prompt: str) -> bool:
        """Determine if the query is generic or specific."""
        few_shot_prompt = self._build_few_shot_prompt(prompt, self.config.context_examples, "web search")
        response = self.llm_manager.call_llm(few_shot_prompt)
        return "needs context" not in response.lower()

    def normalize_url(self, url: str) -> str:
        """Normalize a URL for consistent storage or comparison."""
        normalized_url = (
            url.replace("https://", "")
            .replace("www.", "")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        # print("Normalized URL", normalized_url)
        return normalized_url

    def check_robots_txt(self, urls: List[str]) -> List[str]:
        """Check robots.txt to filter out disallowed URLs."""
        allowed_urls = []
        for url in urls:
            try:
                robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
                rp = RobotFileParser(robots_url)
                rp.read()
                if rp.can_fetch("*", url):
                    allowed_urls.append(url)
            except Exception:
                # If robots.txt is missing or there's an error, assume URL is allowed
                allowed_urls.append(url)
        return allowed_urls

    def is_website_request(self, prompt: str) -> Tuple[List[str], bool]:
        """Detect URLs or domains in the prompt."""
        # Find URLs with protocol
        urls_with_protocol = re.findall(self.config.url_pattern_with_protocol, prompt)
        
        # Find potential domains, filter out those already in full URLs
        all_domains = re.findall(self.config.url_pattern_no_protocol, prompt)
        urls_no_protocol = [
            domain for domain in all_domains
            if not any(domain in url for url in urls_with_protocol)
        ]
        
        # Combine both lists
        urls = urls_with_protocol + urls_no_protocol
        return (urls, bool(urls))

    def input_needed(self, prompt: str) -> bool:
        pass

    def analyze(self, prompt: str) -> bool:
        pass

    def enhance(self, prompt: str) -> bool:
        pass
    
# Example usage
if __name__ == "__main__":
    config = UtilConfig()
    
    util_manager = UtilManager(config)
    
    prompt = "Check https://example.com and google.com for info"
    urls, is_website = util_manager.is_website_request(prompt)
    print(f"URLs: {urls}, Is Website: {is_website}")
    
    allowed_urls = util_manager.check_robots_txt(urls)
    print(f"Allowed URLs: {allowed_urls}")
    
    normalized = util_manager.normalize_url(urls[0])
    print(f"Normalized URL: {normalized}")
    
    needs_web = util_manager.few_shot_needs_web_search("What is the latest news?")
    print(f"Needs web search: {needs_web}")
    
    needs_ctx = util_manager.needs_context("What is the capital of France?")
    print(f"Needs context: {needs_ctx}")