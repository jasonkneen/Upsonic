"""
Crawlee Web Scraping & Crawling Toolkit for Upsonic Framework.

This module provides comprehensive Crawlee integration with support for:
- Scraping single URLs into clean text (HTTP-based, no browser)
- Extracting all links from a page
- Extracting elements using CSS selectors
- Extracting structured table data from HTML tables
- Extracting page metadata (title, meta tags, Open Graph data)
- Crawling entire websites with depth and page limit controls
- Scraping JavaScript-rendered pages via Playwright browser
- Taking screenshots of web pages via Playwright browser

No API key is required. All operations run locally.

Required Packages:
-----------------
- crawlee[beautifulsoup]: For HTTP-based scraping tools (scrape_url, extract_links, etc.)
- crawlee[playwright]: For browser-based tools (scrape_dynamic_page, take_screenshot)
  After installing, run: playwright install

Install Commands:
    pip install 'crawlee[beautifulsoup]'
    pip install 'crawlee[playwright]' && playwright install

Example Usage:
    ```python
    from upsonic.tools.custom_tools.crawlee import CrawleeTools

    tools = CrawleeTools()

    # Scrape a single URL (HTTP, fast)
    result = tools.scrape_url("https://example.com")

    # Extract all links from a page
    result = tools.extract_links("https://example.com")

    # Extract elements by CSS selector
    result = tools.extract_with_selector("https://example.com", selector="h2.title")

    # Extract tables from a page
    result = tools.extract_tables("https://en.wikipedia.org/wiki/Python_(programming_language)")

    # Get page metadata (title, description, OG tags)
    result = tools.get_page_metadata("https://example.com")

    # Crawl a website (multi-page)
    result = tools.crawl_website("https://example.com", max_pages=10, max_depth=2)

    # Scrape a JS-rendered page (browser)
    result = tools.scrape_dynamic_page("https://example.com/spa")

    # Take a screenshot of a page (browser)
    result = tools.take_screenshot("https://example.com")
    ```
"""

import asyncio
import base64
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from upsonic.utils.printing import error_log

try:
    from crawlee.crawlers import (
        BeautifulSoupCrawler,
        BeautifulSoupCrawlingContext,
    )

    _CRAWLEE_BS_AVAILABLE = True
except ImportError:
    BeautifulSoupCrawler = None  # type: ignore[assignment, misc]
    BeautifulSoupCrawlingContext = None  # type: ignore[assignment, misc]
    _CRAWLEE_BS_AVAILABLE = False

try:
    from crawlee.crawlers import (
        PlaywrightCrawler,
        PlaywrightCrawlingContext,
    )

    _CRAWLEE_PW_AVAILABLE = True
except ImportError:
    PlaywrightCrawler = None  # type: ignore[assignment, misc]
    PlaywrightCrawlingContext = None  # type: ignore[assignment, misc]
    _CRAWLEE_PW_AVAILABLE = False


def _run_async(coro: Any) -> Any:
    """Run an async coroutine safely from a sync context.

    Handles the case where an event loop is already running (e.g. when
    the framework wraps a sync tool call) by executing the coroutine in
    a dedicated thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


class CrawleeTools:
    """
    Crawlee web scraping and crawling toolkit.

    Provides HTTP-based (BeautifulSoup) and browser-based (Playwright) tools
    for scraping, crawling, and extracting data from websites. All operations
    run locally with no API key required.

    Attributes:
        headless: Whether to run Playwright browsers in headless mode.
        browser_type: Playwright browser engine ('chromium', 'firefox', 'webkit').
        max_request_retries: Default retry count for failed requests.
        max_concurrency: Maximum number of concurrent requests during crawls.
        proxy_urls: Optional list of proxy server URLs for rotation.
        respect_robots_txt: Whether to honour robots.txt rules.
        max_content_length: Maximum character length of extracted text per page.
    """

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        max_request_retries: int = 3,
        max_concurrency: int = 5,
        proxy_urls: Optional[List[str]] = None,
        respect_robots_txt: bool = True,
        max_content_length: int = 50_000,
        enable_scrape: bool = True,
        enable_extract_links: bool = True,
        enable_extract_with_selector: bool = True,
        enable_extract_tables: bool = True,
        enable_get_page_metadata: bool = True,
        enable_crawl: bool = True,
        enable_scrape_dynamic: bool = True,
        enable_screenshot: bool = True,
        all: bool = False,
    ) -> None:
        """
        Initialize the CrawleeTools toolkit.

        Args:
            headless: Run Playwright browsers in headless mode.
            browser_type: Browser engine for Playwright ('chromium', 'firefox', 'webkit').
            max_request_retries: Number of retries for failed requests.
            max_concurrency: Maximum parallel requests during crawls.
            proxy_urls: List of proxy URLs for proxy rotation.
            respect_robots_txt: Honour robots.txt directives.
            max_content_length: Max characters of extracted text per page.
            enable_scrape: Enable the scrape_url tool.
            enable_extract_links: Enable the extract_links tool.
            enable_extract_with_selector: Enable the extract_with_selector tool.
            enable_extract_tables: Enable the extract_tables tool.
            enable_get_page_metadata: Enable the get_page_metadata tool.
            enable_crawl: Enable the crawl_website tool.
            enable_scrape_dynamic: Enable the scrape_dynamic_page tool.
            enable_screenshot: Enable the take_screenshot tool.
            all: Enable all tools regardless of individual flags.
        """
        if not _CRAWLEE_BS_AVAILABLE and not _CRAWLEE_PW_AVAILABLE:
            from upsonic.utils.printing import import_error

            import_error(
                package_name="crawlee[beautifulsoup]",
                install_command="pip install 'crawlee[beautifulsoup]' 'crawlee[playwright]' && playwright install",
                feature_name="Crawlee tools",
            )

        self.headless: bool = headless
        self.browser_type: str = browser_type
        self.max_request_retries: int = max_request_retries
        self.max_concurrency: int = max_concurrency
        self.proxy_urls: Optional[List[str]] = proxy_urls
        self.respect_robots_txt: bool = respect_robots_txt
        self.max_content_length: int = max_content_length

        self._proxy_config: Optional[Any] = None
        if self.proxy_urls:
            try:
                from crawlee.proxy_configuration import ProxyConfiguration

                self._proxy_config = ProxyConfiguration(
                    proxy_urls=self.proxy_urls,
                )
            except ImportError:
                pass

        self._tools: List[Callable[..., str]] = []

        if enable_scrape or all:
            self._tools.append(self.scrape_url)
        if enable_extract_links or all:
            self._tools.append(self.extract_links)
        if enable_extract_with_selector or all:
            self._tools.append(self.extract_with_selector)
        if enable_extract_tables or all:
            self._tools.append(self.extract_tables)
        if enable_get_page_metadata or all:
            self._tools.append(self.get_page_metadata)
        if enable_crawl or all:
            self._tools.append(self.crawl_website)
        if enable_scrape_dynamic or all:
            self._tools.append(self.scrape_dynamic_page)
        if enable_screenshot or all:
            self._tools.append(self.take_screenshot)

    def functions(self) -> List[Callable[..., str]]:
        """Return the list of enabled tool functions."""
        return self._tools


    def _make_configuration(self) -> Any:
        """Create a Crawlee Configuration that stores data in a temp directory."""
        from crawlee.configuration import Configuration

        tmp_dir = tempfile.mkdtemp(prefix="crawlee_storage_")
        return Configuration(
            storage_dir=tmp_dir,
            purge_on_start=True,
        )

    def _bs_crawler_kwargs(
        self,
        max_requests: int = 1,
    ) -> Dict[str, Any]:
        """Build keyword arguments for a BeautifulSoupCrawler."""
        kwargs: Dict[str, Any] = {
            "configuration": self._make_configuration(),
            "configure_logging": False,
            "max_requests_per_crawl": max_requests,
            "max_request_retries": self.max_request_retries,
            "respect_robots_txt_file": self.respect_robots_txt,
        }
        if self._proxy_config is not None:
            kwargs["proxy_configuration"] = self._proxy_config
        return kwargs

    def _pw_crawler_kwargs(
        self,
        max_requests: int = 1,
    ) -> Dict[str, Any]:
        """Build keyword arguments for a PlaywrightCrawler."""
        kwargs: Dict[str, Any] = {
            "configuration": self._make_configuration(),
            "configure_logging": False,
            "headless": self.headless,
            "browser_type": self.browser_type,
            "max_requests_per_crawl": max_requests,
            "max_request_retries": self.max_request_retries,
            "respect_robots_txt_file": self.respect_robots_txt,
        }
        if self._proxy_config is not None:
            kwargs["proxy_configuration"] = self._proxy_config
        return kwargs


    def scrape_url(
        self,
        url: str,
        only_main_content: bool = True,
        max_content_length: Optional[int] = None,
    ) -> str:
        """
        Scrape a single URL and extract its text content using HTTP (no browser needed).

        Args:
            url: The URL to scrape.
            only_main_content: If True, strips nav/header/footer/script/style elements.
            max_content_length: Max characters to return. Defaults to instance setting.

        Returns:
            JSON string with url, title, text content, and status_code.
        """
        return _run_async(
            self._ascrape_url(url, only_main_content, max_content_length)
        )

    async def _ascrape_url(
        self,
        url: str,
        only_main_content: bool = True,
        max_content_length: Optional[int] = None,
    ) -> str:
        """Async implementation of scrape_url."""
        if not _CRAWLEE_BS_AVAILABLE:
            return json.dumps({"error": "crawlee[beautifulsoup] is not installed."})

        truncate_at: int = max_content_length or self.max_content_length
        results: List[Dict[str, Any]] = []

        crawler = BeautifulSoupCrawler(**self._bs_crawler_kwargs(max_requests=1))

        @crawler.router.default_handler
        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            soup = context.soup
            if only_main_content:
                for tag in soup(
                    ["script", "style", "nav", "footer", "header", "aside", "noscript"]
                ):
                    tag.decompose()

            text: str = soup.get_text(separator="\n", strip=True)
            if len(text) > truncate_at:
                text = text[:truncate_at] + "\n... [truncated]"

            results.append(
                {
                    "url": context.request.url,
                    "title": soup.title.string if soup.title else None,
                    "text": text,
                    "status_code": context.http_response.status_code,
                }
            )

        try:
            await crawler.run([url])
            if results:
                return json.dumps(results[0], default=str)
            return json.dumps({"error": "No data collected", "url": url})
        except Exception as e:
            error_log(f"Crawlee scrape_url error: {e}")
            return json.dumps({"error": str(e), "url": url})


    def extract_links(
        self,
        url: str,
        css_filter: Optional[str] = None,
    ) -> str:
        """
        Extract all links from a web page.

        Args:
            url: The URL to extract links from.
            css_filter: Optional CSS selector to limit which <a> tags are included.

        Returns:
            JSON string with the list of links (href, text) and total count.
        """
        return _run_async(self._aextract_links(url, css_filter))

    async def _aextract_links(
        self,
        url: str,
        css_filter: Optional[str] = None,
    ) -> str:
        """Async implementation of extract_links."""
        if not _CRAWLEE_BS_AVAILABLE:
            return json.dumps({"error": "crawlee[beautifulsoup] is not installed."})

        results: List[Dict[str, Any]] = []

        crawler = BeautifulSoupCrawler(**self._bs_crawler_kwargs(max_requests=1))

        @crawler.router.default_handler
        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            if css_filter:
                a_tags = context.soup.select(css_filter)
                a_tags = [tag for tag in a_tags if tag.name == "a" and tag.get("href")]
                if not a_tags:
                    a_tags = [
                        a
                        for container in context.soup.select(css_filter)
                        for a in container.find_all("a", href=True)
                    ]
            else:
                a_tags = context.soup.find_all("a", href=True)

            links: List[Dict[str, Optional[str]]] = []
            for a_tag in a_tags:
                links.append(
                    {
                        "href": a_tag.get("href"),
                        "text": a_tag.get_text(strip=True) or None,
                    }
                )

            results.append(
                {
                    "url": context.request.url,
                    "links": links,
                    "total_links": len(links),
                    "status_code": context.http_response.status_code,
                }
            )

        try:
            await crawler.run([url])
            if results:
                return json.dumps(results[0], default=str)
            return json.dumps({"error": "No data collected", "url": url})
        except Exception as e:
            error_log(f"Crawlee extract_links error: {e}")
            return json.dumps({"error": str(e), "url": url})


    def extract_with_selector(
        self,
        url: str,
        selector: str,
        max_content_length: Optional[int] = None,
    ) -> str:
        """
        Extract specific elements from a web page using a CSS selector.

        Args:
            url: The URL to scrape.
            selector: CSS selector to match elements (e.g. 'h2.title', 'div.product-card').
            max_content_length: Max characters per element's text. Defaults to instance setting.

        Returns:
            JSON string with matched elements (text, html, tag name, attributes) and count.
        """
        return _run_async(
            self._aextract_with_selector(url, selector, max_content_length)
        )

    async def _aextract_with_selector(
        self,
        url: str,
        selector: str,
        max_content_length: Optional[int] = None,
    ) -> str:
        """Async implementation of extract_with_selector."""
        if not _CRAWLEE_BS_AVAILABLE:
            return json.dumps({"error": "crawlee[beautifulsoup] is not installed."})

        truncate_at: int = max_content_length or self.max_content_length
        results: List[Dict[str, Any]] = []

        crawler = BeautifulSoupCrawler(**self._bs_crawler_kwargs(max_requests=1))

        @crawler.router.default_handler
        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            elements = context.soup.select(selector)
            extracted: List[Dict[str, Any]] = []

            for el in elements:
                text: str = el.get_text(strip=True)
                if len(text) > truncate_at:
                    text = text[:truncate_at] + "... [truncated]"

                html_str: str = str(el)
                if len(html_str) > truncate_at:
                    html_str = html_str[:truncate_at] + "... [truncated]"

                extracted.append(
                    {
                        "text": text,
                        "html": html_str,
                        "tag": el.name,
                        "attributes": {
                            k: (v if isinstance(v, str) else " ".join(v))
                            for k, v in el.attrs.items()
                        },
                    }
                )

            results.append(
                {
                    "url": context.request.url,
                    "selector": selector,
                    "matches": extracted,
                    "count": len(extracted),
                    "status_code": context.http_response.status_code,
                }
            )

        try:
            await crawler.run([url])
            if results:
                return json.dumps(results[0], default=str)
            return json.dumps({"error": "No data collected", "url": url})
        except Exception as e:
            error_log(f"Crawlee extract_with_selector error: {e}")
            return json.dumps({"error": str(e), "url": url})


    def extract_tables(
        self,
        url: str,
        table_index: Optional[int] = None,
    ) -> str:
        """
        Extract HTML tables from a web page as structured JSON.

        Args:
            url: The URL containing tables to extract.
            table_index: If specified, only extract the table at this zero-based index.

        Returns:
            JSON string with table data (headers, rows) for each table found.
        """
        return _run_async(self._aextract_tables(url, table_index))

    async def _aextract_tables(
        self,
        url: str,
        table_index: Optional[int] = None,
    ) -> str:
        """Async implementation of extract_tables."""
        if not _CRAWLEE_BS_AVAILABLE:
            return json.dumps({"error": "crawlee[beautifulsoup] is not installed."})

        results: List[Dict[str, Any]] = []

        crawler = BeautifulSoupCrawler(**self._bs_crawler_kwargs(max_requests=1))

        @crawler.router.default_handler
        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            all_tables = context.soup.find_all("table")
            if table_index is not None:
                if 0 <= table_index < len(all_tables):
                    all_tables = [all_tables[table_index]]
                else:
                    results.append(
                        {
                            "url": context.request.url,
                            "error": f"Table index {table_index} out of range (found {len(all_tables)} tables)",
                            "table_count": len(all_tables),
                        }
                    )
                    return

            tables: List[Dict[str, Any]] = []
            for idx, table in enumerate(all_tables):
                headers: List[str] = []
                thead = table.find("thead")
                if thead:
                    headers = [
                        th.get_text(strip=True) for th in thead.find_all(["th", "td"])
                    ]

                rows: List[List[str]] = []
                tbody = table.find("tbody") or table
                for tr in tbody.find_all("tr"):
                    cells = [
                        td.get_text(strip=True) for td in tr.find_all(["td", "th"])
                    ]
                    if cells and cells != headers:
                        rows.append(cells)

                tables.append(
                    {
                        "table_index": table_index if table_index is not None else idx,
                        "headers": headers,
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": len(headers) if headers else (len(rows[0]) if rows else 0),
                    }
                )

            results.append(
                {
                    "url": context.request.url,
                    "tables": tables,
                    "table_count": len(tables),
                    "status_code": context.http_response.status_code,
                }
            )

        try:
            await crawler.run([url])
            if results:
                return json.dumps(results[0], default=str)
            return json.dumps({"error": "No data collected", "url": url})
        except Exception as e:
            error_log(f"Crawlee extract_tables error: {e}")
            return json.dumps({"error": str(e), "url": url})


    def get_page_metadata(
        self,
        url: str,
    ) -> str:
        """
        Extract metadata from a web page including title, meta tags, and Open Graph data.

        Args:
            url: The URL to extract metadata from.

        Returns:
            JSON string with title, description, canonical URL, Open Graph tags,
            and all meta tags found on the page.
        """
        return _run_async(self._aget_page_metadata(url))

    async def _aget_page_metadata(
        self,
        url: str,
    ) -> str:
        """Async implementation of get_page_metadata."""
        if not _CRAWLEE_BS_AVAILABLE:
            return json.dumps({"error": "crawlee[beautifulsoup] is not installed."})

        results: List[Dict[str, Any]] = []

        crawler = BeautifulSoupCrawler(**self._bs_crawler_kwargs(max_requests=1))

        @crawler.router.default_handler
        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            soup = context.soup

            meta_tags: Dict[str, str] = {}
            for meta in soup.find_all("meta"):
                name = meta.get("name") or meta.get("property")
                content = meta.get("content")
                if name and content:
                    meta_tags[str(name)] = str(content)

            canonical: Optional[str] = None
            link_canonical = soup.find("link", rel="canonical")
            if link_canonical:
                canonical = str(link_canonical.get("href"))

            favicon: Optional[str] = None
            for rel_val in (["icon"], ["shortcut", "icon"]):
                link_icon = soup.find("link", rel=rel_val)
                if link_icon:
                    favicon = str(link_icon.get("href"))
                    break

            results.append(
                {
                    "url": context.request.url,
                    "title": soup.title.string.strip() if soup.title and soup.title.string else None,
                    "description": meta_tags.get("description"),
                    "canonical": canonical,
                    "favicon": favicon,
                    "og_title": meta_tags.get("og:title"),
                    "og_description": meta_tags.get("og:description"),
                    "og_image": meta_tags.get("og:image"),
                    "og_type": meta_tags.get("og:type"),
                    "og_url": meta_tags.get("og:url"),
                    "twitter_card": meta_tags.get("twitter:card"),
                    "twitter_title": meta_tags.get("twitter:title"),
                    "twitter_description": meta_tags.get("twitter:description"),
                    "twitter_image": meta_tags.get("twitter:image"),
                    "all_meta": meta_tags,
                    "status_code": context.http_response.status_code,
                }
            )

        try:
            await crawler.run([url])
            if results:
                return json.dumps(results[0], default=str)
            return json.dumps({"error": "No data collected", "url": url})
        except Exception as e:
            error_log(f"Crawlee get_page_metadata error: {e}")
            return json.dumps({"error": str(e), "url": url})


    def crawl_website(
        self,
        url: str,
        max_pages: int = 10,
        max_depth: Optional[int] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        only_main_content: bool = True,
        max_content_length: Optional[int] = None,
    ) -> str:
        """
        Crawl a website starting from a seed URL, following links across pages.

        Follows same-domain links by default. Returns extracted text from each page.

        Args:
            url: The starting URL to crawl from.
            max_pages: Maximum number of pages to visit.
            max_depth: Maximum link depth from the starting URL (None = unlimited).
            include_patterns: Glob patterns for URLs to include (e.g. ['/blog/**']).
            exclude_patterns: Glob patterns for URLs to exclude (e.g. ['/admin/**']).
            only_main_content: Strip nav/header/footer/script/style elements.
            max_content_length: Max characters of text per page. Defaults to instance setting.

        Returns:
            JSON string with a list of crawled pages (url, title, text) and total count.
        """
        return _run_async(
            self._acrawl_website(
                url,
                max_pages,
                max_depth,
                include_patterns,
                exclude_patterns,
                only_main_content,
                max_content_length,
            )
        )

    async def _acrawl_website(
        self,
        url: str,
        max_pages: int = 10,
        max_depth: Optional[int] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        only_main_content: bool = True,
        max_content_length: Optional[int] = None,
    ) -> str:
        """Async implementation of crawl_website."""
        if not _CRAWLEE_BS_AVAILABLE:
            return json.dumps({"error": "crawlee[beautifulsoup] is not installed."})

        truncate_at: int = max_content_length or self.max_content_length
        pages: List[Dict[str, Any]] = []

        crawler_kwargs = self._bs_crawler_kwargs(max_requests=max_pages)
        if max_depth is not None:
            crawler_kwargs["max_crawl_depth"] = max_depth

        from crawlee import ConcurrencySettings

        crawler_kwargs["concurrency_settings"] = ConcurrencySettings(
            max_concurrency=self.max_concurrency,
            desired_concurrency=min(self.max_concurrency, 10),
        )

        crawler = BeautifulSoupCrawler(**crawler_kwargs)

        @crawler.router.default_handler
        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            soup = context.soup
            if only_main_content:
                for tag in soup(
                    ["script", "style", "nav", "footer", "header", "aside", "noscript"]
                ):
                    tag.decompose()

            text: str = soup.get_text(separator="\n", strip=True)
            if len(text) > truncate_at:
                text = text[:truncate_at] + "\n... [truncated]"

            pages.append(
                {
                    "url": context.request.url,
                    "title": soup.title.string.strip() if soup.title and soup.title.string else None,
                    "text": text,
                    "status_code": context.http_response.status_code,
                }
            )

            enqueue_kwargs: Dict[str, Any] = {"strategy": "same-hostname"}

            if include_patterns:
                try:
                    from crawlee import Glob

                    enqueue_kwargs["include"] = [
                        Glob(p) for p in include_patterns
                    ]
                except ImportError:
                    pass

            if exclude_patterns:
                try:
                    from crawlee import Glob

                    enqueue_kwargs["exclude"] = [
                        Glob(p) for p in exclude_patterns
                    ]
                except ImportError:
                    pass

            await context.enqueue_links(**enqueue_kwargs)

        try:
            await crawler.run([url])
            return json.dumps(
                {
                    "seed_url": url,
                    "pages": pages,
                    "pages_crawled": len(pages),
                },
                default=str,
            )
        except Exception as e:
            error_log(f"Crawlee crawl_website error: {e}")
            return json.dumps({"error": str(e), "url": url})


    def scrape_dynamic_page(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        max_content_length: Optional[int] = None,
    ) -> str:
        """
        Scrape a JavaScript-rendered page using a real browser (Playwright).

        Use this instead of scrape_url when the page relies on JavaScript to
        render its content (single-page apps, lazy-loaded content, etc.).

        Args:
            url: The URL to scrape.
            wait_for_selector: CSS selector to wait for before extracting content.
            max_content_length: Max characters to return. Defaults to instance setting.

        Returns:
            JSON string with url, title, extracted text, and html_length.
        """
        return _run_async(
            self._ascrape_dynamic_page(url, wait_for_selector, max_content_length)
        )

    async def _ascrape_dynamic_page(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        max_content_length: Optional[int] = None,
    ) -> str:
        """Async implementation of scrape_dynamic_page."""
        if not _CRAWLEE_PW_AVAILABLE:
            return json.dumps(
                {
                    "error": (
                        "crawlee[playwright] is not installed. "
                        "Run: pip install 'crawlee[playwright]' && playwright install"
                    )
                }
            )

        truncate_at: int = max_content_length or self.max_content_length
        results: List[Dict[str, Any]] = []

        crawler = PlaywrightCrawler(**self._pw_crawler_kwargs(max_requests=1))

        @crawler.router.default_handler
        async def handler(context: PlaywrightCrawlingContext) -> None:
            page = context.page

            await page.wait_for_load_state("networkidle")

            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=15_000)

            title: str = await page.title()
            text: str = await page.evaluate("() => document.body.innerText")
            html_content: str = await page.content()

            if len(text) > truncate_at:
                text = text[:truncate_at] + "\n... [truncated]"

            results.append(
                {
                    "url": context.request.url,
                    "title": title,
                    "text": text,
                    "html_length": len(html_content),
                }
            )

        try:
            await crawler.run([url])
            if results:
                return json.dumps(results[0], default=str)
            return json.dumps({"error": "No data collected", "url": url})
        except Exception as e:
            error_log(f"Crawlee scrape_dynamic_page error: {e}")
            return json.dumps({"error": str(e), "url": url})


    def take_screenshot(
        self,
        url: str,
        full_page: bool = True,
        output_path: Optional[str] = None,
        wait_for_selector: Optional[str] = None,
    ) -> str:
        """
        Take a screenshot of a web page using a real browser (Playwright).

        The screenshot is saved to disk and the file path is returned.

        Args:
            url: The URL to screenshot.
            full_page: Capture the entire scrollable page, not just the viewport.
            output_path: File path to save the screenshot. Auto-generated if not provided.
            wait_for_selector: CSS selector to wait for before taking the screenshot.

        Returns:
            JSON string with the file path, page title, and screenshot size in bytes.
        """
        return _run_async(
            self._atake_screenshot(url, full_page, output_path, wait_for_selector)
        )

    async def _atake_screenshot(
        self,
        url: str,
        full_page: bool = True,
        output_path: Optional[str] = None,
        wait_for_selector: Optional[str] = None,
    ) -> str:
        """Async implementation of take_screenshot."""
        if not _CRAWLEE_PW_AVAILABLE:
            return json.dumps(
                {
                    "error": (
                        "crawlee[playwright] is not installed. "
                        "Run: pip install 'crawlee[playwright]' && playwright install"
                    )
                }
            )

        if output_path is None:
            tmp_dir = tempfile.mkdtemp(prefix="crawlee_screenshots_")
            from pathlib import Path
            from urllib.parse import urlparse

            parsed = urlparse(url)
            safe_name = (parsed.netloc + parsed.path).replace("/", "_").strip("_") or "page"
            safe_name = safe_name[:80]
            output_path = str(Path(tmp_dir) / f"{safe_name}.png")

        results: List[Dict[str, Any]] = []

        crawler = PlaywrightCrawler(**self._pw_crawler_kwargs(max_requests=1))

        @crawler.router.default_handler
        async def handler(context: PlaywrightCrawlingContext) -> None:
            page = context.page

            await page.wait_for_load_state("networkidle")

            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=15_000)

            screenshot_bytes: bytes = await page.screenshot(
                path=output_path,
                full_page=full_page,
                type="png",
            )

            title: str = await page.title()

            results.append(
                {
                    "url": context.request.url,
                    "title": title,
                    "screenshot_path": output_path,
                    "screenshot_size_bytes": len(screenshot_bytes),
                }
            )

        try:
            await crawler.run([url])
            if results:
                return json.dumps(results[0], default=str)
            return json.dumps({"error": "No data collected", "url": url})
        except Exception as e:
            error_log(f"Crawlee take_screenshot error: {e}")
            return json.dumps({"error": str(e), "url": url})
