from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from backend.llm import call_llm

def is_generic(prompt):
    """Determine if a web search is needed using few-shot prompting with LLM."""
    few_shot_examples = [
        {
            "question": "Who won the current FIFA World Cup?",
            "label": "needs context"
        },
        {
            "question": "What is the latest stock price of Apple?",
            "label": "needs context"
        },
        {
            "question": "What is the capital of France?",
            "label": "answered directly using LLM"
        },
        {
            "question": "Explain the theory of relativity?",
            "label": "answered directly using LLM"
        },
    ]
    
    few_shot_prompt = """
    You are an intelligent system that determines whether a query requires a web search.
    Based on the given examples, classify the new question accordingly.

    Examples:
    """
    
    for example in few_shot_examples:
        few_shot_prompt += f"\nQuestion: {example['question']}\nLabel: {example['label']}\n"
    
    few_shot_prompt += f"\nQuestion: {prompt}\nLabel: "
    
    response = call_llm(few_shot_prompt, require_search=False)
    
    return "needs context" not in response.lower()

    
# ZSC classification/ Few shot prompting technique addition remaining

def normalize_url(url):
    normalized_url = (
        url.replace("https://", "")
        .replace("www.", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    print("Normalized URL", normalized_url)
    return normalized_url

def check_robots_txt(urls: list[str]) -> list[str]:
    allowed_urls = []

    for url in urls:
        try:
            robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
            rp = RobotFileParser(robots_url)
            rp.read()

            if rp.can_fetch("*", url):
                allowed_urls.append(url)

        except Exception:
            # If robots.txt is missing or there's any error, assume URL is allowed
            allowed_urls.append(url)

    return allowed_urls