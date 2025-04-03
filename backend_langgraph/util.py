from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from backend.llm import call_llm
import re

def few_shot_needs_web_search(prompt):
    """Determine if a web search is needed using few-shot prompting with LLM."""
    few_shot_examples = [
        {
            "question": "Who won the current/latest FIFA World Cup?",
            "label": "needs web search"
        },
        {
            "question": "What is the latest stock price of Apple?",
            "label": "needs web search"
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
    You are an intelligent system that determines whether a query requires a web search or not.
    Based on the given examples, classify the new question accordingly.
    Strictly give only the labels as the response specified in the examples above.

    Examples:
    """
    
    for example in few_shot_examples:
        few_shot_prompt += f"\nQuestion: {example['question']}\nLabel: {example['label']}\n"
    
    few_shot_prompt += f"\nQuestion: {prompt}\nLabel: "
    
    response = call_llm(few_shot_prompt, require_search=False)
    print(response)
    return "needs web" in response.lower()

def needs_context(prompt):
    """Determine if the query is generic or specific."""
    few_shot_examples = [
        {
            "question": "Who won the current/latest FIFA World Cup?",
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
    Strictly give only the labels as the response specified in the examples above.

    Examples:
    """
    
    for example in few_shot_examples:
        few_shot_prompt += f"\nQuestion: {example['question']}\nLabel: {example['label']}\n"
    
    few_shot_prompt += f"\nQuestion: {prompt}\nLabel: "
    
    response = call_llm(few_shot_prompt, require_search=False)
    return "needs context" not in response.lower()
    

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

def is_valid_website(prompt: str) -> bool:
    # Implement logic to check if prompt is a valid website URL.
    # For now, return True if the prompt starts with "http" (as an example)
    return prompt.startswith("http")

def is_website_request(prompt):
    url_pattern_with_protocol = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Pattern for domain names without protocol
    url_pattern_no_protocol = r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[a-zA-Z0-9-._~:/?#[\]@!$&\'()*+,;=]*)?'
    
    # Find all matches with protocol
    urls_with_protocol = re.findall(url_pattern_with_protocol, prompt)
    
    # Find all potential domains, then filter out ones that are part of full URLs
    all_domains = re.findall(url_pattern_no_protocol, prompt)
    urls_no_protocol = []
    
    for domain in all_domains:
        # Check if this domain is part of a URL with protocol
        if not any(domain in url for url in urls_with_protocol):
            urls_no_protocol.append(domain)
    
    # Combine both lists
    urls = urls_with_protocol + urls_no_protocol
    
    return (urls, bool(urls))
