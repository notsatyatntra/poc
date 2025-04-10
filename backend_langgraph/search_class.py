from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from backend_langgraph.util_class import UtilManager, UtilConfig
import asyncio

class SearchTool(ABC):
    """Abstract base class for web search tools."""
    
    @abstractmethod
    async def search(self, query: str, max_results: int) -> List[str]:
        """Perform a web search and return a list of URLs."""
        pass

class DuckDuckGoSearchTool(SearchTool):
    """DuckDuckGo search tool implementation."""
    
    def __init__(self, exclude_domains: List[str] = None):
        self.exclude_domains = exclude_domains or ["youtube.com", "britannica.com", "vimeo.com"]
    
    async def search(self, query: str, max_results: int) -> List[str]:
        try:
            search_query = query
            for domain in self.exclude_domains:
                search_query += f" -site:{domain}"
            results = DDGS().text(search_query, max_results=max_results)
            return [result["href"] for result in results]
        except Exception as e:
            print(f"❌ Failed to fetch results from DuckDuckGo: {str(e)}")
            return []

class SearchConfig:
    """Configuration class for search and crawl settings."""
    
    def __init__(
        self,
        search_tool: SearchTool = None,
        num_results: int = 10,
        exclude_domains: Optional[List[str]] = None,
        bm25_threshold: float = 1.2,
        crawler_excluded_tags: List[str] = ["nav", "footer", "header", "form", "img", "a"],
        crawler_only_text: bool = True,
        crawler_exclude_social_media: bool = True,
        crawler_keep_data_attributes: bool = False,
        crawler_cache_mode: CacheMode = CacheMode.BYPASS,
        crawler_remove_overlay_elements: bool = True,
        crawler_user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        crawler_page_timeout: int = 20000,  # in ms
        browser_headless: bool = True,
        browser_text_mode: bool = True,
        browser_light_mode: bool = True,
        util_config: Optional[UtilConfig] = None
    ):
        """
        Args:
            search_tool (SearchTool): The search tool to use (e.g., DuckDuckGoSearchTool).
            num_results (int): Number of search results to fetch.
            exclude_domains (List[str]): Domains to exclude from search results.
            bm25_threshold (float): BM25 threshold for content filtering.
            crawler_excluded_tags (List[str]): Tags to exclude during crawling.
            crawler_only_text (bool): Whether to extract only text.
            crawler_exclude_social_media (bool): Whether to exclude social media links.
            crawler_keep_data_attributes (bool): Whether to keep data attributes.
            crawler_cache_mode (CacheMode): Cache mode for crawling.
            crawler_remove_overlay_elements (bool): Whether to remove overlay elements.
            crawler_user_agent (str): User agent for the crawler.
            crawler_page_timeout (int): Page timeout in milliseconds.
            browser_headless (bool): Run browser in headless mode.
            browser_text_mode (bool): Use text-only mode in browser.
            browser_light_mode (bool): Use light mode in browser.
            util_config (UtilConfig): Configuration for UtilManager.
        """
        self.search_tool = search_tool or DuckDuckGoSearchTool(exclude_domains=exclude_domains)
        self.num_results = num_results
        self.bm25_threshold = bm25_threshold
        self.crawler_excluded_tags = crawler_excluded_tags
        self.crawler_only_text = crawler_only_text
        self.crawler_exclude_social_media = crawler_exclude_social_media
        self.crawler_keep_data_attributes = crawler_keep_data_attributes
        self.crawler_cache_mode = crawler_cache_mode
        self.crawler_remove_overlay_elements = crawler_remove_overlay_elements
        self.crawler_user_agent = crawler_user_agent
        self.crawler_page_timeout = crawler_page_timeout
        self.browser_headless = browser_headless
        self.browser_text_mode = browser_text_mode
        self.browser_light_mode = browser_light_mode
        self.util_config = util_config or UtilConfig()

class SearchManager:
    """Manages web search and crawling operations."""
    
    def __init__(self, config: SearchConfig):
        """
        Initialize the SearchManager with a configuration.

        Args:
            config (SearchConfig): Configuration object for search and crawl settings.
        """
        self.config = config
        self.util_manager = UtilManager(self.config.util_config)  # For robots.txt checking

    async def get_web_urls(self, search_term: str) -> List[str]:
        """Perform a web search and return a list of filtered URLs."""
        urls = await self.config.search_tool.search(search_term, self.config.num_results)
        return self.util_manager.check_robots_txt(urls)

    async def crawl_webpages(self, urls: List[str], prompt: str):
        """Crawl webpages asynchronously and return crawl results."""
        bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=self.config.bm25_threshold)
        md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

        crawler_config = CrawlerRunConfig(
            markdown_generator=md_generator,
            excluded_tags=self.config.crawler_excluded_tags,
            only_text=self.config.crawler_only_text,
            exclude_social_media_links=self.config.crawler_exclude_social_media,
            keep_data_attributes=self.config.crawler_keep_data_attributes,
            cache_mode=self.config.crawler_cache_mode,
            remove_overlay_elements=self.config.crawler_remove_overlay_elements,
            user_agent=self.config.crawler_user_agent,
            page_timeout=self.config.crawler_page_timeout,
        )
        browser_config = BrowserConfig(
            headless=self.config.browser_headless,
            text_mode=self.config.browser_text_mode,
            light_mode=self.config.browser_light_mode
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            results = await crawler.arun_many(urls, config=crawler_config)
            return results

    async def search_and_crawl(self, query: str, urls: List[str] = None) -> List:
        """Perform web search, crawl pages, and return results."""
        if urls:
            return await self.crawl_webpages(urls, query)
        urls = await self.get_web_urls(query)
        return await self.crawl_webpages(urls, query)

# Example usage
if __name__ == "__main__":
    async def main():
        # Define a configuration (customize as needed)
        config = SearchConfig(
            search_tool=DuckDuckGoSearchTool(),
            num_results=5,
            bm25_threshold=1.5,
            crawler_page_timeout=30000  # 30 seconds
        )
        
        # Initialize the search manager
        search_manager = SearchManager(config)
        
        # Test search and crawl
        query = "latest AI advancements"
        results = await search_manager.search_and_crawl(query)
        print(f"Crawl Results: {results}")
        
        # Test with specific URLs
        specific_urls = ["https://example.com", "https://ai.googleblog.com"]
        results = await search_manager.search_and_crawl(query, urls=specific_urls)
        print(f"Crawl Results for specific URLs: {results}")

    asyncio.run(main())