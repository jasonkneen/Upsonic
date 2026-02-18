"""
Firecrawl Web Scraping & Crawling Toolkit for Upsonic Framework.

This module provides comprehensive Firecrawl API integration with support for:
- Scraping single URLs into markdown, HTML, or structured JSON
- Crawling entire websites with configurable depth and limits
- Mapping website URLs for discovery
- Searching the web with content scraping
- Batch scraping multiple URLs simultaneously
- Extracting structured data using LLM-powered extraction
- Async operations with job management (start, status, cancel)

Required Environment Variables:
-----------------------------
- FIRECRAWL_API_KEY: Firecrawl API key from https://firecrawl.dev

How to Get API Key:
------------------
1. Go to https://firecrawl.dev
2. Sign up for an account
3. Navigate to your dashboard
4. Copy your API key

Example Usage:
    ```python
    from upsonic.tools.custom_tools.firecrawl import FirecrawlTools

    tools = FirecrawlTools(api_key="fc-YOUR-API-KEY")

    # Scrape a single URL
    result = await tools.scrape_url("https://example.com")

    # Crawl a website
    result = await tools.crawl_website("https://example.com", limit=10)

    # Search the web
    result = await tools.search_web("AI agent frameworks", limit=5)

    # Extract structured data
    result = await tools.extract_data(
        urls=["https://example.com"],
        prompt="Extract the company description",
    )
    ```
"""

import json
from os import getenv
from typing import Any, Dict, List, Optional, Union

from upsonic.utils.printing import error_log

try:
    from firecrawl import AsyncFirecrawl, Firecrawl
    _FIRECRAWL_AVAILABLE = True
except ImportError:
    AsyncFirecrawl = None
    Firecrawl = None
    _FIRECRAWL_AVAILABLE = False


def _serialize(result: Any) -> str:
    """Serialize a Firecrawl SDK response to a JSON string.

    firecrawl-py v4 returns Pydantic BaseModel objects. This helper calls
    model_dump() when available so json.dumps receives a plain dict/list
    instead of a Pydantic object (which would otherwise fall through to
    the default=str path and produce an opaque string representation).
    """
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), default=str)
    return json.dumps(result, default=str)


class FirecrawlTools:
    """
    Firecrawl web scraping, crawling, and data extraction toolkit.

    This toolkit provides methods for:
    - Scraping single URLs with multiple output formats
    - Crawling entire websites with depth/limit controls
    - Mapping website URL structures
    - Searching the web with optional content scraping
    - Batch scraping multiple URLs simultaneously
    - Extracting structured data via LLM-powered extraction
    - Managing async crawl/batch/extract jobs

    Attributes:
        api_key: Firecrawl API key
        api_url: Custom API base URL (for self-hosted instances)
        default_formats: Default output formats for scraping
        default_scrape_limit: Default page limit for crawl operations
        default_search_limit: Default result limit for search operations
        timeout: Default timeout for blocking operations in seconds
        poll_interval: Default poll interval for job status checks in seconds
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        default_formats: Optional[List[str]] = None,
        default_scrape_limit: int = 100,
        default_search_limit: int = 5,
        timeout: int = 120,
        poll_interval: int = 2,
        enable_scrape: bool = True,
        enable_crawl: bool = True,
        enable_map: bool = True,
        enable_search: bool = True,
        enable_batch_scrape: bool = True,
        enable_extract: bool = True,
        enable_crawl_management: bool = True,
        enable_batch_management: bool = True,
        enable_extract_management: bool = True,
        all: bool = False,
    ) -> None:
        """
        Initialize the FirecrawlTools toolkit.

        Args:
            api_key: Firecrawl API key. Falls back to FIRECRAWL_API_KEY env var.
            api_url: Custom API base URL for self-hosted Firecrawl instances.
            default_formats: Default output formats for scrape operations.
            default_scrape_limit: Default page limit for crawl operations.
            default_search_limit: Default result limit for search operations.
            timeout: Default timeout for blocking operations in seconds.
            poll_interval: Default poll interval for job status checks in seconds.
            enable_scrape: Enable scrape_url tool.
            enable_crawl: Enable crawl_website and start_crawl tools.
            enable_map: Enable map_website tool.
            enable_search: Enable search_web tool.
            enable_batch_scrape: Enable batch_scrape and start_batch_scrape tools.
            enable_extract: Enable extract_data and start_extract tools.
            enable_crawl_management: Enable get_crawl_status and cancel_crawl tools.
            enable_batch_management: Enable get_batch_scrape_status tool.
            enable_extract_management: Enable get_extract_status tool.
            all: Enable all tools regardless of individual flags.
        """
        if not _FIRECRAWL_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="firecrawl-py",
                install_command='pip install firecrawl-py',
                feature_name="Firecrawl tools"
            )

        self.api_key: str = api_key or getenv("FIRECRAWL_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Firecrawl API key is required. Set FIRECRAWL_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.api_url: Optional[str] = api_url
        self.default_formats: List[str] = default_formats or ["markdown"]
        self.default_scrape_limit: int = default_scrape_limit
        self.default_search_limit: int = default_search_limit
        self.timeout: int = timeout
        self.poll_interval: int = poll_interval

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.api_url:
            client_kwargs["api_url"] = self.api_url

        self.sync_client: Firecrawl = Firecrawl(**client_kwargs)
        self.async_client: AsyncFirecrawl = AsyncFirecrawl(**client_kwargs)

        self._tools: List[Any] = []
        if enable_scrape or all:
            self._tools.append(self.scrape_url)
        if enable_crawl or all:
            self._tools.append(self.crawl_website)
            self._tools.append(self.start_crawl)
        if enable_map or all:
            self._tools.append(self.map_website)
        if enable_search or all:
            self._tools.append(self.search_web)
        if enable_batch_scrape or all:
            self._tools.append(self.batch_scrape)
            self._tools.append(self.start_batch_scrape)
        if enable_extract or all:
            self._tools.append(self.extract_data)
            self._tools.append(self.start_extract)
        if enable_crawl_management or all:
            self._tools.append(self.get_crawl_status)
            self._tools.append(self.cancel_crawl)
        if enable_batch_management or all:
            self._tools.append(self.get_batch_scrape_status)
        if enable_extract_management or all:
            self._tools.append(self.get_extract_status)

    def functions(self) -> List[Any]:
        """Return the list of enabled tool functions."""
        return self._tools

    # ─── Scraping ─────────────────────────────────────────────────────

    def scrape_url(
        self,
        url: str,
        formats: Optional[List[str]] = None,
        only_main_content: Optional[bool] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        wait_for: Optional[int] = None,
        timeout: Optional[int] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        json_prompt: Optional[str] = None,
        location: Optional[str] = None,
        mobile: Optional[bool] = None,
        skip_tls_verification: Optional[bool] = None,
        remove_base64_images: Optional[bool] = None,
    ) -> str:
        """
        Scrape a single URL and extract its content.

        Args:
            url: The URL to scrape.
            formats: Output formats (markdown, html, rawHtml, links, summary, images).
            only_main_content: Extract only the main content, excluding headers/footers/navs.
            include_tags: HTML tags to include in extraction.
            exclude_tags: HTML tags to exclude from extraction.
            wait_for: Time in ms to wait for page to load before scraping.
            timeout: Timeout in ms for the scrape operation.
            json_schema: JSON schema for structured data extraction via LLM.
            json_prompt: Prompt for LLM-based JSON extraction.
            location: Country/location for geo-targeted scraping.
            mobile: Scrape as mobile device.
            skip_tls_verification: Skip TLS certificate verification.
            remove_base64_images: Remove base64 images from output.

        Returns:
            JSON string containing the scraped content.
        """
        try:
            scrape_formats: List[Any] = list(formats or self.default_formats)

            if json_schema:
                json_format: Dict[str, Any] = {"type": "json", "schema": json_schema}
                if json_prompt:
                    json_format["prompt"] = json_prompt
                scrape_formats.append(json_format)

            kwargs: Dict[str, Any] = {"formats": scrape_formats}

            if only_main_content is not None:
                kwargs["only_main_content"] = only_main_content
            if include_tags is not None:
                kwargs["include_tags"] = include_tags
            if exclude_tags is not None:
                kwargs["exclude_tags"] = exclude_tags
            if wait_for is not None:
                kwargs["wait_for"] = wait_for
            if timeout is not None:
                kwargs["timeout"] = timeout
            if location is not None:
                kwargs["location"] = location
            if mobile is not None:
                kwargs["mobile"] = mobile
            if skip_tls_verification is not None:
                kwargs["skip_tls_verification"] = skip_tls_verification
            if remove_base64_images is not None:
                kwargs["remove_base64_images"] = remove_base64_images

            result = self.sync_client.scrape(url, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl scrape error: {e}")
            return json.dumps({"error": str(e)})

    async def _ascrape_url(
        self,
        url: str,
        formats: Optional[List[str]] = None,
        only_main_content: Optional[bool] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        wait_for: Optional[int] = None,
        timeout: Optional[int] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        json_prompt: Optional[str] = None,
        location: Optional[str] = None,
        mobile: Optional[bool] = None,
        skip_tls_verification: Optional[bool] = None,
        remove_base64_images: Optional[bool] = None,
    ) -> str:
        """
        Async version of scrape_url. Scrape a single URL and extract its content.

        Args:
            url: The URL to scrape.
            formats: Output formats (markdown, html, rawHtml, links, summary, images).
            only_main_content: Extract only the main content, excluding headers/footers/navs.
            include_tags: HTML tags to include in extraction.
            exclude_tags: HTML tags to exclude from extraction.
            wait_for: Time in ms to wait for page to load before scraping.
            timeout: Timeout in ms for the scrape operation.
            json_schema: JSON schema for structured data extraction via LLM.
            json_prompt: Prompt for LLM-based JSON extraction.
            location: Country/location for geo-targeted scraping.
            mobile: Scrape as mobile device.
            skip_tls_verification: Skip TLS certificate verification.
            remove_base64_images: Remove base64 images from output.

        Returns:
            JSON string containing the scraped content.
        """
        try:
            scrape_formats: List[Any] = list(formats or self.default_formats)

            if json_schema:
                json_format: Dict[str, Any] = {"type": "json", "schema": json_schema}
                if json_prompt:
                    json_format["prompt"] = json_prompt
                scrape_formats.append(json_format)

            kwargs: Dict[str, Any] = {"formats": scrape_formats}

            if only_main_content is not None:
                kwargs["only_main_content"] = only_main_content
            if include_tags is not None:
                kwargs["include_tags"] = include_tags
            if exclude_tags is not None:
                kwargs["exclude_tags"] = exclude_tags
            if wait_for is not None:
                kwargs["wait_for"] = wait_for
            if timeout is not None:
                kwargs["timeout"] = timeout
            if location is not None:
                kwargs["location"] = location
            if mobile is not None:
                kwargs["mobile"] = mobile
            if skip_tls_verification is not None:
                kwargs["skip_tls_verification"] = skip_tls_verification
            if remove_base64_images is not None:
                kwargs["remove_base64_images"] = remove_base64_images

            result = await self.async_client.scrape(url, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async scrape error: {e}")
            return json.dumps({"error": str(e)})

    # ─── Crawling ─────────────────────────────────────────────────────

    def crawl_website(
        self,
        url: str,
        limit: Optional[int] = None,
        scrape_formats: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None,
        max_discovery_depth: Optional[int] = None,
        sitemap: Optional[str] = None,
        poll_interval: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Crawl an entire website (blocking). Waits for the crawl to complete.

        Args:
            url: The starting URL to crawl.
            limit: Maximum number of pages to crawl.
            scrape_formats: Output formats for scraped pages.
            exclude_paths: URL path patterns to exclude.
            include_paths: URL path patterns to include.
            max_discovery_depth: Maximum crawl depth from the starting URL.
            sitemap: Sitemap mode ('skip', 'include', or 'only').
            poll_interval: Polling interval in seconds for job status checks.
            timeout: Timeout in seconds for the entire crawl operation.

        Returns:
            JSON string containing the crawl results.
        """
        try:
            kwargs: Dict[str, Any] = {}

            kwargs["limit"] = limit or self.default_scrape_limit
            kwargs["poll_interval"] = poll_interval or self.poll_interval
            kwargs["timeout"] = timeout or self.timeout

            if scrape_formats:
                kwargs["scrape_options"] = {"formats": scrape_formats}
            if exclude_paths is not None:
                kwargs["exclude_paths"] = exclude_paths
            if include_paths is not None:
                kwargs["include_paths"] = include_paths
            if max_discovery_depth is not None:
                kwargs["max_discovery_depth"] = max_discovery_depth
            if sitemap is not None:
                kwargs["sitemap"] = sitemap

            result = self.sync_client.crawl(url=url, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl crawl error: {e}")
            return json.dumps({"error": str(e)})

    async def _acrawl_website(
        self,
        url: str,
        limit: Optional[int] = None,
        scrape_formats: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None,
        max_discovery_depth: Optional[int] = None,
        sitemap: Optional[str] = None,
        poll_interval: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Async version of crawl_website. Crawl an entire website (blocking).

        Args:
            url: The starting URL to crawl.
            limit: Maximum number of pages to crawl.
            scrape_formats: Output formats for scraped pages.
            exclude_paths: URL path patterns to exclude.
            include_paths: URL path patterns to include.
            max_discovery_depth: Maximum crawl depth from the starting URL.
            sitemap: Sitemap mode ('skip', 'include', or 'only').
            poll_interval: Polling interval in seconds for job status checks.
            timeout: Timeout in seconds for the entire crawl operation.

        Returns:
            JSON string containing the crawl results.
        """
        try:
            kwargs: Dict[str, Any] = {}

            kwargs["limit"] = limit or self.default_scrape_limit
            kwargs["poll_interval"] = poll_interval or self.poll_interval
            kwargs["timeout"] = timeout or self.timeout

            if scrape_formats:
                kwargs["scrape_options"] = {"formats": scrape_formats}
            if exclude_paths is not None:
                kwargs["exclude_paths"] = exclude_paths
            if include_paths is not None:
                kwargs["include_paths"] = include_paths
            if max_discovery_depth is not None:
                kwargs["max_discovery_depth"] = max_discovery_depth
            if sitemap is not None:
                kwargs["sitemap"] = sitemap

            result = await self.async_client.crawl(url=url, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async crawl error: {e}")
            return json.dumps({"error": str(e)})

    def start_crawl(
        self,
        url: str,
        limit: Optional[int] = None,
        scrape_formats: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None,
        max_discovery_depth: Optional[int] = None,
        sitemap: Optional[str] = None,
    ) -> str:
        """
        Start a non-blocking crawl job. Returns a job ID for status tracking.

        Args:
            url: The starting URL to crawl.
            limit: Maximum number of pages to crawl.
            scrape_formats: Output formats for scraped pages.
            exclude_paths: URL path patterns to exclude.
            include_paths: URL path patterns to include.
            max_discovery_depth: Maximum crawl depth from the starting URL.
            sitemap: Sitemap mode ('skip', 'include', or 'only').

        Returns:
            JSON string containing the job ID and initial status.
        """
        try:
            kwargs: Dict[str, Any] = {}

            kwargs["limit"] = limit or self.default_scrape_limit

            if scrape_formats:
                kwargs["scrape_options"] = {"formats": scrape_formats}
            if exclude_paths is not None:
                kwargs["exclude_paths"] = exclude_paths
            if include_paths is not None:
                kwargs["include_paths"] = include_paths
            if max_discovery_depth is not None:
                kwargs["max_discovery_depth"] = max_discovery_depth
            if sitemap is not None:
                kwargs["sitemap"] = sitemap

            result = self.sync_client.start_crawl(url=url, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl start_crawl error: {e}")
            return json.dumps({"error": str(e)})

    async def _astart_crawl(
        self,
        url: str,
        limit: Optional[int] = None,
        scrape_formats: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None,
        max_discovery_depth: Optional[int] = None,
        sitemap: Optional[str] = None,
    ) -> str:
        """
        Async version of start_crawl. Start a non-blocking crawl job.

        Args:
            url: The starting URL to crawl.
            limit: Maximum number of pages to crawl.
            scrape_formats: Output formats for scraped pages.
            exclude_paths: URL path patterns to exclude.
            include_paths: URL path patterns to include.
            max_discovery_depth: Maximum crawl depth from the starting URL.
            sitemap: Sitemap mode ('skip', 'include', or 'only').

        Returns:
            JSON string containing the job ID and initial status.
        """
        try:
            kwargs: Dict[str, Any] = {}

            kwargs["limit"] = limit or self.default_scrape_limit

            if scrape_formats:
                kwargs["scrape_options"] = {"formats": scrape_formats}
            if exclude_paths is not None:
                kwargs["exclude_paths"] = exclude_paths
            if include_paths is not None:
                kwargs["include_paths"] = include_paths
            if max_discovery_depth is not None:
                kwargs["max_discovery_depth"] = max_discovery_depth
            if sitemap is not None:
                kwargs["sitemap"] = sitemap

            result = await self.async_client.start_crawl(url=url, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async start_crawl error: {e}")
            return json.dumps({"error": str(e)})

    def get_crawl_status(self, job_id: str) -> str:
        """
        Check the status of an active crawl job.

        Args:
            job_id: The crawl job ID returned by start_crawl.

        Returns:
            JSON string containing the crawl status, completed pages, and data.
        """
        try:
            result = self.sync_client.get_crawl_status(job_id)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl get_crawl_status error: {e}")
            return json.dumps({"error": str(e)})

    async def _aget_crawl_status(self, job_id: str) -> str:
        """
        Async version of get_crawl_status.

        Args:
            job_id: The crawl job ID returned by start_crawl.

        Returns:
            JSON string containing the crawl status, completed pages, and data.
        """
        try:
            result = await self.async_client.get_crawl_status(job_id)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async get_crawl_status error: {e}")
            return json.dumps({"error": str(e)})

    def cancel_crawl(self, job_id: str) -> str:
        """
        Cancel an ongoing crawl job.

        Args:
            job_id: The crawl job ID to cancel.

        Returns:
            JSON string indicating cancellation success.
        """
        try:
            result = self.sync_client.cancel_crawl(job_id)
            return json.dumps({"cancelled": result}, default=str)
        except Exception as e:
            error_log(f"Firecrawl cancel_crawl error: {e}")
            return json.dumps({"error": str(e)})

    async def _acancel_crawl(self, job_id: str) -> str:
        """
        Async version of cancel_crawl.

        Args:
            job_id: The crawl job ID to cancel.

        Returns:
            JSON string indicating cancellation success.
        """
        try:
            result = await self.async_client.cancel_crawl(job_id)
            return json.dumps({"cancelled": result}, default=str)
        except Exception as e:
            error_log(f"Firecrawl async cancel_crawl error: {e}")
            return json.dumps({"error": str(e)})

    # ─── Mapping ──────────────────────────────────────────────────────

    def map_website(
        self,
        url: str,
        limit: Optional[int] = None,
    ) -> str:
        """
        Generate a list of URLs from a website for discovery.

        Args:
            url: The website URL to map.
            limit: Maximum number of URLs to return.

        Returns:
            JSON string containing the list of discovered URLs.
        """
        try:
            kwargs: Dict[str, Any] = {}
            if limit is not None:
                kwargs["limit"] = limit

            result = self.sync_client.map(url=url, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl map error: {e}")
            return json.dumps({"error": str(e)})

    async def _amap_website(
        self,
        url: str,
        limit: Optional[int] = None,
    ) -> str:
        """
        Async version of map_website.

        Args:
            url: The website URL to map.
            limit: Maximum number of URLs to return.

        Returns:
            JSON string containing the list of discovered URLs.
        """
        try:
            kwargs: Dict[str, Any] = {}
            if limit is not None:
                kwargs["limit"] = limit

            result = await self.async_client.map(url=url, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async map error: {e}")
            return json.dumps({"error": str(e)})

    # ─── Searching ────────────────────────────────────────────────────

    def search_web(
        self,
        query: str,
        limit: Optional[int] = None,
        scrape_options: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None,
        tbs: Optional[str] = None,
        timeout: Optional[int] = None,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> str:
        """
        Search the web and optionally scrape search result content.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            scrape_options: Options for scraping search results (e.g. {"formats": ["markdown"]}).
            location: Country/region for geo-targeted search results.
            tbs: Time-based search filter (e.g. 'qdr:d' for past day, 'qdr:w' for past week).
            timeout: Timeout in milliseconds for the search operation.
            sources: Result types to include ('web', 'news', 'images').
            categories: Filter by categories ('pdf', 'research', 'github').

        Returns:
            JSON string containing the search results.
        """
        try:
            kwargs: Dict[str, Any] = {"query": query}

            kwargs["limit"] = limit or self.default_search_limit

            if scrape_options is not None:
                kwargs["scrape_options"] = scrape_options
            if location is not None:
                kwargs["location"] = location
            if tbs is not None:
                kwargs["tbs"] = tbs
            if timeout is not None:
                kwargs["timeout"] = timeout
            if sources is not None:
                kwargs["sources"] = sources
            if categories is not None:
                kwargs["categories"] = categories

            result = self.sync_client.search(**kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl search error: {e}")
            return json.dumps({"error": str(e)})

    async def _asearch_web(
        self,
        query: str,
        limit: Optional[int] = None,
        scrape_options: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None,
        tbs: Optional[str] = None,
        timeout: Optional[int] = None,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> str:
        """
        Async version of search_web.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            scrape_options: Options for scraping search results (e.g. {"formats": ["markdown"]}).
            location: Country/region for geo-targeted search results.
            tbs: Time-based search filter (e.g. 'qdr:d' for past day, 'qdr:w' for past week).
            timeout: Timeout in milliseconds for the search operation.
            sources: Result types to include ('web', 'news', 'images').
            categories: Filter by categories ('pdf', 'research', 'github').

        Returns:
            JSON string containing the search results.
        """
        try:
            kwargs: Dict[str, Any] = {"query": query}

            kwargs["limit"] = limit or self.default_search_limit

            if scrape_options is not None:
                kwargs["scrape_options"] = scrape_options
            if location is not None:
                kwargs["location"] = location
            if tbs is not None:
                kwargs["tbs"] = tbs
            if timeout is not None:
                kwargs["timeout"] = timeout
            if sources is not None:
                kwargs["sources"] = sources
            if categories is not None:
                kwargs["categories"] = categories

            result = await self.async_client.search(**kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async search error: {e}")
            return json.dumps({"error": str(e)})

    # ─── Batch Scraping ───────────────────────────────────────────────

    def batch_scrape(
        self,
        urls: List[str],
        formats: Optional[List[str]] = None,
        poll_interval: Optional[int] = None,
    ) -> str:
        """
        Batch scrape multiple URLs (blocking). Waits for all results.

        Args:
            urls: List of URLs to scrape.
            formats: Output formats for the scraped content.
            poll_interval: Polling interval in seconds for status checks.

        Returns:
            JSON string containing the batch scrape results.
        """
        try:
            kwargs: Dict[str, Any] = {}

            kwargs["formats"] = formats or self.default_formats
            kwargs["poll_interval"] = poll_interval or self.poll_interval

            result = self.sync_client.batch_scrape(urls, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl batch_scrape error: {e}")
            return json.dumps({"error": str(e)})

    async def _abatch_scrape(
        self,
        urls: List[str],
        formats: Optional[List[str]] = None,
        poll_interval: Optional[int] = None,
    ) -> str:
        """
        Async version of batch_scrape.

        Args:
            urls: List of URLs to scrape.
            formats: Output formats for the scraped content.
            poll_interval: Polling interval in seconds for status checks.

        Returns:
            JSON string containing the batch scrape results.
        """
        try:
            kwargs: Dict[str, Any] = {}

            kwargs["formats"] = formats or self.default_formats
            kwargs["poll_interval"] = poll_interval or self.poll_interval

            result = await self.async_client.batch_scrape(urls, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async batch_scrape error: {e}")
            return json.dumps({"error": str(e)})

    def start_batch_scrape(
        self,
        urls: List[str],
        formats: Optional[List[str]] = None,
    ) -> str:
        """
        Start a non-blocking batch scrape job. Returns a job ID for tracking.

        Args:
            urls: List of URLs to scrape.
            formats: Output formats for the scraped content.

        Returns:
            JSON string containing the job ID and initial status.
        """
        try:
            kwargs: Dict[str, Any] = {}
            kwargs["formats"] = formats or self.default_formats

            result = self.sync_client.start_batch_scrape(urls, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl start_batch_scrape error: {e}")
            return json.dumps({"error": str(e)})

    async def _astart_batch_scrape(
        self,
        urls: List[str],
        formats: Optional[List[str]] = None,
    ) -> str:
        """
        Async version of start_batch_scrape.

        Args:
            urls: List of URLs to scrape.
            formats: Output formats for the scraped content.

        Returns:
            JSON string containing the job ID and initial status.
        """
        try:
            kwargs: Dict[str, Any] = {}
            kwargs["formats"] = formats or self.default_formats

            result = await self.async_client.start_batch_scrape(urls, **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async start_batch_scrape error: {e}")
            return json.dumps({"error": str(e)})

    def get_batch_scrape_status(self, job_id: str) -> str:
        """
        Check the status of a batch scrape job.

        Args:
            job_id: The batch scrape job ID returned by start_batch_scrape.

        Returns:
            JSON string containing the batch status, completed count, and data.
        """
        try:
            result = self.sync_client.get_batch_scrape_status(job_id)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl get_batch_scrape_status error: {e}")
            return json.dumps({"error": str(e)})

    async def _aget_batch_scrape_status(self, job_id: str) -> str:
        """
        Async version of get_batch_scrape_status.

        Args:
            job_id: The batch scrape job ID returned by start_batch_scrape.

        Returns:
            JSON string containing the batch status, completed count, and data.
        """
        try:
            result = await self.async_client.get_batch_scrape_status(job_id)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async get_batch_scrape_status error: {e}")
            return json.dumps({"error": str(e)})

    # ─── Extraction ───────────────────────────────────────────────────

    def extract_data(
        self,
        urls: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        enable_web_search: Optional[bool] = None,
        scrape_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Extract structured data from URLs using LLM-powered extraction (blocking).

        Args:
            urls: List of URLs to extract from. Supports wildcards (e.g. 'example.com/*').
            prompt: Natural language prompt describing what data to extract.
            schema: JSON schema defining the structure of data to extract.
            enable_web_search: Allow extraction to follow links outside specified domains.
            scrape_options: Additional scrape options for the extraction.

        Returns:
            JSON string containing the extracted structured data.
        """
        try:
            kwargs: Dict[str, Any] = {}

            if urls is not None:
                kwargs["urls"] = urls
            if prompt is not None:
                kwargs["prompt"] = prompt
            if schema is not None:
                kwargs["schema"] = schema
            if enable_web_search is not None:
                kwargs["enable_web_search"] = enable_web_search
            if scrape_options is not None:
                from firecrawl.types import ScrapeOptions as _ScrapeOptions
                kwargs["scrape_options"] = _ScrapeOptions(**scrape_options) if isinstance(scrape_options, dict) else scrape_options

            result = self.sync_client.extract(**kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl extract error: {e}")
            return json.dumps({"error": str(e)})

    async def _aextract_data(
        self,
        urls: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        enable_web_search: Optional[bool] = None,
        scrape_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Async version of extract_data.

        Args:
            urls: List of URLs to extract from. Supports wildcards (e.g. 'example.com/*').
            prompt: Natural language prompt describing what data to extract.
            schema: JSON schema defining the structure of data to extract.
            enable_web_search: Allow extraction to follow links outside specified domains.
            scrape_options: Additional scrape options for the extraction.

        Returns:
            JSON string containing the extracted structured data.
        """
        try:
            kwargs: Dict[str, Any] = {}

            if urls is not None:
                kwargs["urls"] = urls
            if prompt is not None:
                kwargs["prompt"] = prompt
            if schema is not None:
                kwargs["schema"] = schema
            if enable_web_search is not None:
                kwargs["enable_web_search"] = enable_web_search
            if scrape_options is not None:
                from firecrawl.types import ScrapeOptions as _ScrapeOptions
                kwargs["scrape_options"] = _ScrapeOptions(**scrape_options) if isinstance(scrape_options, dict) else scrape_options

            result = await self.async_client.extract(**kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async extract error: {e}")
            return json.dumps({"error": str(e)})

    def start_extract(
        self,
        urls: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        enable_web_search: Optional[bool] = None,
    ) -> str:
        """
        Start a non-blocking extraction job. Returns a job ID for tracking.

        Args:
            urls: List of URLs to extract from. Supports wildcards.
            prompt: Natural language prompt describing what data to extract.
            schema: JSON schema defining the structure of data to extract.
            enable_web_search: Allow extraction to follow links outside specified domains.

        Returns:
            JSON string containing the job ID and initial status.
        """
        try:
            kwargs: Dict[str, Any] = {}

            if prompt is not None:
                kwargs["prompt"] = prompt
            if schema is not None:
                kwargs["schema"] = schema
            if enable_web_search is not None:
                kwargs["enable_web_search"] = enable_web_search

            result = self.sync_client.start_extract(urls or [], **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl start_extract error: {e}")
            return json.dumps({"error": str(e)})

    async def _astart_extract(
        self,
        urls: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        enable_web_search: Optional[bool] = None,
    ) -> str:
        """
        Async version of start_extract.

        Args:
            urls: List of URLs to extract from. Supports wildcards.
            prompt: Natural language prompt describing what data to extract.
            schema: JSON schema defining the structure of data to extract.
            enable_web_search: Allow extraction to follow links outside specified domains.

        Returns:
            JSON string containing the job ID and initial status.
        """
        try:
            kwargs: Dict[str, Any] = {}

            if prompt is not None:
                kwargs["prompt"] = prompt
            if schema is not None:
                kwargs["schema"] = schema
            if enable_web_search is not None:
                kwargs["enable_web_search"] = enable_web_search

            result = await self.async_client.start_extract(urls or [], **kwargs)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async start_extract error: {e}")
            return json.dumps({"error": str(e)})

    def get_extract_status(self, job_id: str) -> str:
        """
        Check the status of an extraction job.

        Args:
            job_id: The extraction job ID returned by start_extract.

        Returns:
            JSON string containing the extraction status and data.
        """
        try:
            result = self.sync_client.get_extract_status(job_id)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl get_extract_status error: {e}")
            return json.dumps({"error": str(e)})

    async def _aget_extract_status(self, job_id: str) -> str:
        """
        Async version of get_extract_status.

        Args:
            job_id: The extraction job ID returned by start_extract.

        Returns:
            JSON string containing the extraction status and data.
        """
        try:
            result = await self.async_client.get_extract_status(job_id)
            return _serialize(result)
        except Exception as e:
            error_log(f"Firecrawl async get_extract_status error: {e}")
            return json.dumps({"error": str(e)})
