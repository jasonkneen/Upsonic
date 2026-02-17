"""
Smoke tests for FirecrawlTools — real API requests against Firecrawl.

These tests require a valid FIRECRAWL_API_KEY environment variable.
They are automatically skipped when the key is not set.

Success criteria:
- Every sync and async method on FirecrawlTools returns valid JSON
- Scraping, crawling, mapping, searching, batch scraping, extraction, and
  job-management endpoints all work end-to-end against the live API
- The functions() enablement flags work correctly
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import pytest

FIRECRAWL_API_KEY: Optional[str] = os.getenv("FIRECRAWL_API_KEY")

pytestmark = [
    pytest.mark.skipif(
        not FIRECRAWL_API_KEY,
        reason="FIRECRAWL_API_KEY not set; skipping Firecrawl smoke tests",
    ),
    pytest.mark.timeout(300),
]

TEST_URL: str = "https://example.com"
TEST_SEARCH_QUERY: str = "Python programming language"


@pytest.fixture(scope="module")
def tools() -> Any:
    from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
    return FirecrawlTools(api_key=FIRECRAWL_API_KEY, all=True)


def _parse(raw: str) -> Dict[str, Any]:
    """Parse a JSON string returned by FirecrawlTools methods."""
    parsed: Any = json.loads(raw)
    if isinstance(parsed, dict) and parsed.get("error"):
        pytest.fail(f"API returned error: {parsed['error']}")
    assert isinstance(parsed, dict), (
        f"Expected dict, got {type(parsed).__name__}: {str(parsed)[:200]}"
    )
    return parsed


# ──────────────────────────────────────────────────────────────────────
#  Initialization & functions()
# ──────────────────────────────────────────────────────────────────────

class TestInit:

    def test_init_with_api_key(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)
        assert t.api_key == FIRECRAWL_API_KEY

    def test_init_defaults(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)
        assert t.default_formats == ["markdown"]
        assert t.default_scrape_limit == 100
        assert t.default_search_limit == 5
        assert t.timeout == 120
        assert t.poll_interval == 2

    def test_init_custom_config(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            default_formats=["markdown", "html"],
            default_scrape_limit=50,
            default_search_limit=10,
            timeout=60,
            poll_interval=5,
        )
        assert t.default_formats == ["markdown", "html"]
        assert t.default_scrape_limit == 50
        assert t.default_search_limit == 10
        assert t.timeout == 60
        assert t.poll_interval == 5

    def test_init_missing_key_raises(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        from unittest.mock import patch as _patch
        with _patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Firecrawl API key is required"):
                FirecrawlTools(api_key="")


class TestFunctions:

    def test_default_functions(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)
        names: List[str] = [f.__name__ for f in t.functions()]
        assert "scrape_url" in names
        assert "crawl_website" in names
        assert "start_crawl" in names
        assert "map_website" in names
        assert "search_web" in names

    def test_all_functions_enabled(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(api_key=FIRECRAWL_API_KEY, all=True)
        names: List[str] = [f.__name__ for f in t.functions()]
        expected: List[str] = [
            "scrape_url", "crawl_website", "start_crawl",
            "map_website", "search_web",
            "batch_scrape", "start_batch_scrape",
            "extract_data", "start_extract",
            "get_crawl_status", "cancel_crawl",
            "get_batch_scrape_status", "get_extract_status",
        ]
        for name in expected:
            assert name in names, f"{name} missing from functions()"

    def test_selective_enablement(self) -> None:
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        t: FirecrawlTools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            enable_scrape=True,
            enable_crawl=False,
            enable_map=False,
            enable_search=False,
            enable_batch_scrape=False,
            enable_extract=False,
            enable_crawl_management=False,
            enable_batch_management=False,
            enable_extract_management=False,
        )
        names: List[str] = [f.__name__ for f in t.functions()]
        assert "scrape_url" in names
        assert "crawl_website" not in names
        assert "search_web" not in names


# ──────────────────────────────────────────────────────────────────────
#  Scraping (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestScrapeUrl:

    def test_scrape_url_basic(self, tools: Any) -> None:
        result: str = tools.scrape_url(TEST_URL)
        parsed: Dict[str, Any] = _parse(result)
        assert "markdown" in parsed or "metadata" in parsed

    def test_scrape_url_with_formats(self, tools: Any) -> None:
        result: str = tools.scrape_url(TEST_URL, formats=["markdown", "html"])
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_scrape_url_only_main_content(self, tools: Any) -> None:
        result: str = tools.scrape_url(TEST_URL, only_main_content=True)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_scrape_url_with_tags(self, tools: Any) -> None:
        result: str = tools.scrape_url(
            TEST_URL,
            include_tags=["h1", "p"],
            exclude_tags=["nav", "footer"],
        )
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_scrape_url_mobile(self, tools: Any) -> None:
        result: str = tools.scrape_url(TEST_URL, mobile=True)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_scrape_url_remove_base64(self, tools: Any) -> None:
        result: str = tools.scrape_url(TEST_URL, remove_base64_images=True)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_scrape_url_returns_valid_json(self, tools: Any) -> None:
        result: str = tools.scrape_url(TEST_URL)
        parsed: Any = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_ascrape_url_basic(self, tools: Any) -> None:
        result: str = await tools._ascrape_url(TEST_URL)
        parsed: Dict[str, Any] = _parse(result)
        assert "markdown" in parsed or "metadata" in parsed

    @pytest.mark.asyncio
    async def test_ascrape_url_with_formats(self, tools: Any) -> None:
        result: str = await tools._ascrape_url(TEST_URL, formats=["markdown", "html"])
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)


# ──────────────────────────────────────────────────────────────────────
#  Crawling — blocking (sync + async)
#  Note: crawl rate limit on free tier is 3 req/min. Tests use sleeps.
# ──────────────────────────────────────────────────────────────────────

class TestCrawlWebsite:

    def test_crawl_website_basic(self, tools: Any) -> None:
        result: str = tools.crawl_website(TEST_URL, limit=2, max_discovery_depth=1)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_acrawl_website_basic(self, tools: Any) -> None:
        result: str = await tools._acrawl_website(TEST_URL, limit=2, max_discovery_depth=1)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)


# ──────────────────────────────────────────────────────────────────────
#  Crawling — non-blocking start / status / cancel
# ──────────────────────────────────────────────────────────────────────

class TestCrawlManagement:

    def test_start_crawl_get_status_and_cancel(self, tools: Any) -> None:
        """Start a crawl, check its status, then cancel it."""
        time.sleep(65)

        start_raw: str = tools.start_crawl(TEST_URL, limit=2, max_discovery_depth=1)
        start_parsed: Dict[str, Any] = _parse(start_raw)
        job_id: str = start_parsed.get("id", "")
        assert job_id, f"No job id returned: {start_parsed}"

        time.sleep(3)

        status_raw: str = tools.get_crawl_status(job_id)
        status_parsed: Dict[str, Any] = _parse(status_raw)
        assert "status" in status_parsed

        cancel_raw: str = tools.cancel_crawl(job_id)
        cancel_parsed: Dict[str, Any] = json.loads(cancel_raw)
        assert isinstance(cancel_parsed, dict)

    @pytest.mark.asyncio
    async def test_astart_crawl_aget_status_and_acancel(self, tools: Any) -> None:
        """Async: start a crawl, check status, then cancel."""
        time.sleep(65)

        start_raw: str = await tools._astart_crawl(TEST_URL, limit=2, max_discovery_depth=1)
        start_parsed: Dict[str, Any] = _parse(start_raw)
        job_id: str = start_parsed.get("id", "")
        assert job_id

        time.sleep(3)

        status_raw: str = await tools._aget_crawl_status(job_id)
        status_parsed: Dict[str, Any] = _parse(status_raw)
        assert "status" in status_parsed

        cancel_raw: str = await tools._acancel_crawl(job_id)
        cancel_parsed: Dict[str, Any] = json.loads(cancel_raw)
        assert isinstance(cancel_parsed, dict)


# ──────────────────────────────────────────────────────────────────────
#  Mapping (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestMapWebsite:

    def test_map_website_basic(self, tools: Any) -> None:
        result: str = tools.map_website(TEST_URL)
        parsed: Dict[str, Any] = _parse(result)
        assert "links" in parsed

    def test_map_website_with_limit(self, tools: Any) -> None:
        result: str = tools.map_website(TEST_URL, limit=5)
        parsed: Dict[str, Any] = _parse(result)
        assert "links" in parsed

    @pytest.mark.asyncio
    async def test_amap_website_basic(self, tools: Any) -> None:
        result: str = await tools._amap_website(TEST_URL)
        parsed: Dict[str, Any] = _parse(result)
        assert "links" in parsed

    @pytest.mark.asyncio
    async def test_amap_website_with_limit(self, tools: Any) -> None:
        result: str = await tools._amap_website(TEST_URL, limit=5)
        parsed: Dict[str, Any] = _parse(result)
        assert "links" in parsed


# ──────────────────────────────────────────────────────────────────────
#  Searching (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestSearchWeb:

    def test_search_web_basic(self, tools: Any) -> None:
        result: str = tools.search_web(TEST_SEARCH_QUERY)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_search_web_with_limit(self, tools: Any) -> None:
        result: str = tools.search_web(TEST_SEARCH_QUERY, limit=3)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_search_web_with_location(self, tools: Any) -> None:
        result: str = tools.search_web(TEST_SEARCH_QUERY, limit=2, location="US")
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_search_web_with_time_filter(self, tools: Any) -> None:
        result: str = tools.search_web(TEST_SEARCH_QUERY, limit=2, tbs="qdr:m")
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_search_web_returns_valid_json(self, tools: Any) -> None:
        result: str = tools.search_web(TEST_SEARCH_QUERY, limit=2)
        parsed: Any = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_asearch_web_basic(self, tools: Any) -> None:
        result: str = await tools._asearch_web(TEST_SEARCH_QUERY, limit=2)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_asearch_web_with_options(self, tools: Any) -> None:
        result: str = await tools._asearch_web(
            TEST_SEARCH_QUERY, limit=2, location="US",
        )
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)


# ──────────────────────────────────────────────────────────────────────
#  Batch scraping — blocking (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestBatchScrape:

    BATCH_URLS: List[str] = [
        "https://example.com",
        "https://example.org",
    ]

    def test_batch_scrape_basic(self, tools: Any) -> None:
        result: str = tools.batch_scrape(self.BATCH_URLS)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_batch_scrape_with_formats(self, tools: Any) -> None:
        result: str = tools.batch_scrape(self.BATCH_URLS, formats=["markdown", "html"])
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_abatch_scrape_basic(self, tools: Any) -> None:
        result: str = await tools._abatch_scrape(self.BATCH_URLS)
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)


# ──────────────────────────────────────────────────────────────────────
#  Batch scraping — non-blocking start / status (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestBatchScrapeManagement:

    BATCH_URLS: List[str] = [
        "https://example.com",
        "https://example.org",
    ]

    def test_start_batch_scrape_and_get_status(self, tools: Any) -> None:
        start_raw: str = tools.start_batch_scrape(self.BATCH_URLS)
        start_parsed: Dict[str, Any] = _parse(start_raw)
        job_id: str = start_parsed.get("id", "")
        assert job_id, f"No job id: {start_parsed}"

        time.sleep(5)
        status_raw: str = tools.get_batch_scrape_status(job_id)
        status_parsed: Dict[str, Any] = _parse(status_raw)
        assert "status" in status_parsed

    @pytest.mark.asyncio
    async def test_astart_batch_scrape_and_aget_status(self, tools: Any) -> None:
        start_raw: str = await tools._astart_batch_scrape(self.BATCH_URLS)
        start_parsed: Dict[str, Any] = _parse(start_raw)
        job_id: str = start_parsed.get("id", "")
        assert job_id

        time.sleep(5)
        status_raw: str = await tools._aget_batch_scrape_status(job_id)
        status_parsed: Dict[str, Any] = _parse(status_raw)
        assert "status" in status_parsed


# ──────────────────────────────────────────────────────────────────────
#  Extraction — blocking (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestExtractData:

    def test_extract_data_with_prompt(self, tools: Any) -> None:
        result: str = tools.extract_data(
            urls=["https://example.com"],
            prompt="Extract the page title and description",
        )
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    def test_extract_data_with_schema(self, tools: Any) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
        }
        result: str = tools.extract_data(
            urls=["https://example.com"],
            prompt="Extract the page title and description",
            schema=schema,
        )
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_aextract_data_with_prompt(self, tools: Any) -> None:
        result: str = await tools._aextract_data(
            urls=["https://example.com"],
            prompt="Extract the page title and description",
        )
        parsed: Dict[str, Any] = _parse(result)
        assert isinstance(parsed, dict)


# ──────────────────────────────────────────────────────────────────────
#  Extraction — non-blocking start / status (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestExtractManagement:

    def test_start_extract_and_get_status(self, tools: Any) -> None:
        start_raw: str = tools.start_extract(
            urls=["https://example.com"],
            prompt="Extract the page title",
        )
        start_parsed: Dict[str, Any] = _parse(start_raw)
        job_id: str = start_parsed.get("id", "")
        assert job_id, f"No job id: {start_parsed}"

        time.sleep(5)
        status_raw: str = tools.get_extract_status(job_id)
        status_parsed: Dict[str, Any] = _parse(status_raw)
        assert "status" in status_parsed

    @pytest.mark.asyncio
    async def test_astart_extract_and_aget_status(self, tools: Any) -> None:
        start_raw: str = await tools._astart_extract(
            urls=["https://example.com"],
            prompt="Extract the page title",
        )
        start_parsed: Dict[str, Any] = _parse(start_raw)
        job_id: str = start_parsed.get("id", "")
        assert job_id

        time.sleep(5)
        status_raw: str = await tools._aget_extract_status(job_id)
        status_parsed: Dict[str, Any] = _parse(status_raw)
        assert "status" in status_parsed


# ──────────────────────────────────────────────────────────────────────
#  JSON serialization sanity
# ──────────────────────────────────────────────────────────────────────

class TestJsonSerialization:

    def test_scrape_returns_valid_json(self, tools: Any) -> None:
        raw: str = tools.scrape_url(TEST_URL)
        parsed: Any = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_search_returns_valid_json(self, tools: Any) -> None:
        raw: str = tools.search_web(TEST_SEARCH_QUERY, limit=2)
        parsed: Any = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_map_returns_valid_json(self, tools: Any) -> None:
        raw: str = tools.map_website(TEST_URL)
        parsed: Any = json.loads(raw)
        assert isinstance(parsed, dict)
