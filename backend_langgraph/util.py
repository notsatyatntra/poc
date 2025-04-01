from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

def needs_web_search(prompt,labels = ["needs context", "answered directly using LLM"]):
    """Determine if a web search is needed based on the query."""
    web_search_keywords = {"what", "who", "when", "where", "how", "latest", "current"}
    prompt_words = set(prompt.lower().split())
    return bool(web_search_keywords & prompt_words)
    


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