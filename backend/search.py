from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from util import check_robots_txt

def get_web_urls(search_term, num_results=10):
    """Perform a web search and return a list of URLs."""
    try:
        discard_urls = ["youtube.com", "britannica.com", "vimeo.com"]
        for url in discard_urls:
            search_term += f" -site:{url}"

        results = DDGS().text(search_term, max_results=num_results)
        results = [result["href"] for result in results]

        return check_robots_txt(results)

    except Exception as e:
        error_msg = ("❌ Failed to fetch results from the web", str(e))
        print(error_msg)

async def crawl_webpages(urls, prompt):
    """Crawl webpages asynchronously and return crawl results."""
    bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=1.2)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        page_timeout=20000,  # in ms: 20 seconds
    )
    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls, config=crawler_config)
        return results

async def search_and_crawl(query, num_results=10):
    """Perform web search, crawl pages, and process into documents."""
    urls = get_web_urls(query, num_results)
    crawl_results = await crawl_webpages(urls, query)
    return crawl_results

# def process_crawl_results(results):
#     """Process crawl results into LangChain Document objects."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
#     documents = []
#     for result in results:
#         if result.markdown_v2:
#             markdown_content = result.markdown_v2.fit_markdown
#             splits = text_splitter.split_text(markdown_content)
#             for split in splits:
#                 documents.append(Document(page_content=split, metadata={"source": result.url}))
#     return documents

