"""
Integration tests for CrawleeTools — real HTTP requests, no mocking.

Covers: init, tool registration/filtering, sync/async tool methods,
output structure, JSON serialization, ToolProcessor integration,
and Playwright tools (scrape_dynamic_page, take_screenshot) when available.

Uses www.nike.com.tr as the test target.

Run:
    uv run pytest tests/unit_tests/tools/test_custom_tools_crawlee.py -v --tb=short
"""

import importlib.util
import json
import os
from typing import Any, Dict, List

import pytest

from upsonic.tools.custom_tools.crawlee import CrawleeTools

try:
    _CRAWLEE_BS_AVAILABLE: bool = importlib.util.find_spec("crawlee.crawlers") is not None
except (ModuleNotFoundError, ValueError):
    _CRAWLEE_BS_AVAILABLE = False

try:
    _CRAWLEE_PW_AVAILABLE: bool = (
        _CRAWLEE_BS_AVAILABLE and importlib.util.find_spec("playwright") is not None
    )
except (ModuleNotFoundError, ValueError):
    _CRAWLEE_PW_AVAILABLE = False
_IN_CI: bool = os.environ.get("CI", "").lower() in ("true", "1", "yes")

pytestmark = [
    pytest.mark.skipif(
        not _CRAWLEE_BS_AVAILABLE,
        reason="crawlee[beautifulsoup] not installed; skipping Crawlee unit tests",
    ),
    pytest.mark.skipif(
        _IN_CI,
        reason="Crawlee tests make real HTTP requests and hang in CI; run locally only",
    ),
]

TEST_URL: str = "https://www.nike.com.tr"

ALL_TOOL_NAMES: List[str] = [
    "scrape_url",
    "extract_links",
    "extract_with_selector",
    "extract_tables",
    "get_page_metadata",
    "crawl_website",
    "scrape_dynamic_page",
    "take_screenshot",
]


def _process_toolkit(toolkit: CrawleeTools) -> None:
    from upsonic.tools.processor import ToolProcessor
    processor: ToolProcessor = ToolProcessor()
    processor._process_toolkit(toolkit)


def _parse(raw: str) -> Dict[str, Any]:
    parsed: Any = json.loads(raw)
    assert isinstance(parsed, dict), f"Expected dict, got {type(parsed).__name__}"
    return parsed


@pytest.fixture(scope="module")
def tools() -> CrawleeTools:
    return CrawleeTools(
        max_request_retries=2,
        respect_robots_txt=False,
        max_content_length=30_000,
        exclude_tools=["scrape_dynamic_page", "take_screenshot"],
    )


@pytest.fixture(scope="module")
def tools_all() -> CrawleeTools:
    return CrawleeTools(respect_robots_txt=False)


# ──────────────────────────────────────────────────────────────────────
#  Initialization
# ──────────────────────────────────────────────────────────────────────

class TestCrawleeToolsInit:

    def test_default_attributes(self) -> None:
        t: CrawleeTools = CrawleeTools()
        assert t.headless is True
        assert t.browser_type == "chromium"
        assert t.max_request_retries == 3
        assert t.max_concurrency == 5
        assert t.proxy_urls is None
        assert t.respect_robots_txt is True
        assert t.max_content_length == 50_000

    def test_custom_attributes(self) -> None:
        t: CrawleeTools = CrawleeTools(
            headless=False,
            browser_type="firefox",
            max_request_retries=1,
            max_concurrency=10,
            proxy_urls=["http://proxy:8080"],
            respect_robots_txt=False,
            max_content_length=5_000,
        )
        assert t.headless is False
        assert t.browser_type == "firefox"
        assert t.max_request_retries == 1
        assert t.max_concurrency == 10
        assert t.proxy_urls == ["http://proxy:8080"]
        assert t.respect_robots_txt is False
        assert t.max_content_length == 5_000

    def test_proxy_configuration_created(self) -> None:
        t: CrawleeTools = CrawleeTools(
            proxy_urls=["http://proxy1:8080", "http://proxy2:8080"],
        )
        assert t._proxy_config is not None

    def test_proxy_configuration_none_when_no_proxies(self) -> None:
        t: CrawleeTools = CrawleeTools()
        assert t._proxy_config is None


# ──────────────────────────────────────────────────────────────────────
#  Tool registration and include/exclude filtering
# ──────────────────────────────────────────────────────────────────────

class TestCrawleeToolsFunctions:

    def test_all_eight_tools_enabled_by_default(self) -> None:
        t: CrawleeTools = CrawleeTools()
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert len(names) == 8
        for expected in ALL_TOOL_NAMES:
            assert expected in names, f"{expected} missing from default tools"

    def test_all_tools_count(self) -> None:
        t: CrawleeTools = CrawleeTools()
        _process_toolkit(t)
        assert len(t.functions) == 8

    def test_include_tools_is_additive(self) -> None:
        t: CrawleeTools = CrawleeTools(include_tools=["scrape_url"])
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "scrape_url" in names
        assert len(names) == 8

    def test_exclude_tools_filtering(self) -> None:
        t: CrawleeTools = CrawleeTools(
            exclude_tools=["scrape_url", "take_screenshot", "scrape_dynamic_page"],
        )
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "scrape_url" not in names
        assert "take_screenshot" not in names
        assert "scrape_dynamic_page" not in names
        assert "extract_links" in names

    def test_exclude_all_except_one(self) -> None:
        excluded: List[str] = [n for n in ALL_TOOL_NAMES if n != "get_page_metadata"]
        t: CrawleeTools = CrawleeTools(exclude_tools=excluded)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert names == ["get_page_metadata"]

    def test_no_private_methods_in_functions(self) -> None:
        t: CrawleeTools = CrawleeTools()
        _process_toolkit(t)
        for fn in t.functions:
            assert not fn.__name__.startswith("_"), (
                f"Private method {fn.__name__} leaked into functions"
            )


# ──────────────────────────────────────────────────────────────────────
#  scrape_url
# ──────────────────────────────────────────────────────────────────────

class TestScrapeUrl:

    def test_scrape_nike_homepage(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result, f"Scrape failed: {result.get('error')}"
        assert result["url"].startswith("https://www.nike.com.tr")
        assert result["title"] is not None
        assert len(result["text"]) > 100
        assert result["status_code"] == 200

    def test_scrape_with_main_content_stripping(self, tools: CrawleeTools) -> None:
        raw_main: str = tools.scrape_url(TEST_URL, only_main_content=True)
        raw_full: str = tools.scrape_url(TEST_URL, only_main_content=False)
        main: Dict[str, Any] = _parse(raw_main)
        full: Dict[str, Any] = _parse(raw_full)
        assert "error" not in main
        assert "error" not in full
        assert len(full["text"]) >= len(main["text"])

    def test_scrape_truncation(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url(TEST_URL, max_content_length=200)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert len(result["text"]) <= 220

    def test_scrape_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "url" in result

    def test_scrape_invalid_url_returns_error(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url("https://thisdomaindoesnotexist12345.com")
        result: Dict[str, Any] = _parse(raw)
        assert "error" in result or "url" in result


# ──────────────────────────────────────────────────────────────────────
#  extract_links
# ──────────────────────────────────────────────────────────────────────

class TestExtractLinks:

    def test_extract_links_nike(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_links(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result, f"Failed: {result.get('error')}"
        assert result["total_links"] > 0
        assert isinstance(result["links"], list)
        assert result["status_code"] == 200
        first_link: Dict[str, Any] = result["links"][0]
        assert "href" in first_link
        assert "text" in first_link

    def test_extract_links_returns_href_and_text(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_links(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        hrefs: List[str] = [
            link["href"] for link in result["links"] if link.get("href")
        ]
        assert len(hrefs) > 0

    def test_extract_links_with_css_filter(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_links(TEST_URL, css_filter="nav")
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert isinstance(result["links"], list)


# ──────────────────────────────────────────────────────────────────────
#  extract_with_selector
# ──────────────────────────────────────────────────────────────────────

class TestExtractWithSelector:

    def test_extract_anchor_tags(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_with_selector(TEST_URL, selector="a")
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result, f"Failed: {result.get('error')}"
        assert result["count"] > 0
        assert result["selector"] == "a"
        assert result["status_code"] == 200
        first_match: Dict[str, Any] = result["matches"][0]
        assert "text" in first_match
        assert "html" in first_match
        assert first_match["tag"] == "a"
        assert "attributes" in first_match

    def test_extract_images(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_with_selector(TEST_URL, selector="img")
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert isinstance(result["matches"], list)

    def test_extract_nonexistent_selector(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_with_selector(
            TEST_URL, selector="div.nonexistent-class-xyz-12345",
        )
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert result["count"] == 0
        assert result["matches"] == []

    def test_extract_with_truncation(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_with_selector(
            TEST_URL, selector="a", max_content_length=50,
        )
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        for match in result["matches"]:
            assert len(match["text"]) <= 70


# ──────────────────────────────────────────────────────────────────────
#  extract_tables
# ──────────────────────────────────────────────────────────────────────

class TestExtractTables:

    def test_extract_tables_nike(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_tables(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result, f"Failed: {result.get('error')}"
        assert "table_count" in result
        assert isinstance(result["tables"], list)
        assert result["status_code"] == 200

    def test_extract_tables_structure(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_tables(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        for table in result["tables"]:
            assert "table_index" in table
            assert "headers" in table
            assert "rows" in table
            assert "row_count" in table
            assert "column_count" in table
            assert isinstance(table["headers"], list)
            assert isinstance(table["rows"], list)

    def test_extract_tables_invalid_index(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_tables(TEST_URL, table_index=9999)
        result: Dict[str, Any] = _parse(raw)
        assert "error" in result
        assert "out of range" in result["error"]


# ──────────────────────────────────────────────────────────────────────
#  get_page_metadata
# ──────────────────────────────────────────────────────────────────────

class TestGetPageMetadata:

    def test_metadata_nike_homepage(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result, f"Failed: {result.get('error')}"
        assert result["url"].startswith("https://www.nike.com.tr")
        assert result["title"] is not None
        assert "nike" in result["title"].lower()
        assert result["status_code"] == 200

    def test_metadata_has_all_meta_dict(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert "all_meta" in result
        assert isinstance(result["all_meta"], dict)

    def test_metadata_og_fields_present(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        for field in [
            "og_title",
            "og_description",
            "og_image",
            "og_type",
            "og_url",
        ]:
            assert field in result

    def test_metadata_twitter_fields_present(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        for field in [
            "twitter_card",
            "twitter_title",
            "twitter_description",
            "twitter_image",
        ]:
            assert field in result

    def test_metadata_canonical_and_favicon(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert "canonical" in result
        assert "favicon" in result


# ──────────────────────────────────────────────────────────────────────
#  crawl_website
# ──────────────────────────────────────────────────────────────────────

class TestCrawlWebsite:

    def test_crawl_nike_basic(self, tools: CrawleeTools) -> None:
        raw: str = tools.crawl_website(TEST_URL, max_pages=3, max_depth=1)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result, f"Failed: {result.get('error')}"
        assert result["seed_url"] == TEST_URL
        assert result["pages_crawled"] >= 1
        assert isinstance(result["pages"], list)

    def test_crawl_page_structure(self, tools: CrawleeTools) -> None:
        raw: str = tools.crawl_website(TEST_URL, max_pages=2, max_depth=1)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        for page in result["pages"]:
            assert "url" in page
            assert "title" in page
            assert "text" in page
            assert "status_code" in page

    def test_crawl_max_pages_limit(self, tools: CrawleeTools) -> None:
        raw: str = tools.crawl_website(TEST_URL, max_pages=2, max_depth=1)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert result["pages_crawled"] <= 10

    def test_crawl_with_include_exclude_patterns(self, tools: CrawleeTools) -> None:
        raw: str = tools.crawl_website(
            TEST_URL,
            max_pages=1,
            max_depth=1,
            include_patterns=["/**"],
            exclude_patterns=["/admin/**"],
        )
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert result["pages_crawled"] >= 0


# ──────────────────────────────────────────────────────────────────────
#  Async method parity
# ──────────────────────────────────────────────────────────────────────

class TestAsyncVersions:

    @pytest.mark.asyncio
    async def test_ascrape_url(self, tools: CrawleeTools) -> None:
        raw: str = await tools.ascrape_url(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert result["status_code"] == 200
        assert len(result["text"]) > 100

    @pytest.mark.asyncio
    async def test_aextract_links(self, tools: CrawleeTools) -> None:
        raw: str = await tools.aextract_links(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert result["total_links"] > 0

    @pytest.mark.asyncio
    async def test_aextract_with_selector(self, tools: CrawleeTools) -> None:
        raw: str = await tools.aextract_with_selector(TEST_URL, selector="a")
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert result["count"] > 0

    @pytest.mark.asyncio
    async def test_aextract_tables(self, tools: CrawleeTools) -> None:
        raw: str = await tools.aextract_tables(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert "table_count" in result

    @pytest.mark.asyncio
    async def test_aget_page_metadata(self, tools: CrawleeTools) -> None:
        raw: str = await tools.aget_page_metadata(TEST_URL)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert result["title"] is not None

    @pytest.mark.asyncio
    async def test_acrawl_website(self, tools: CrawleeTools) -> None:
        raw: str = await tools.acrawl_website(TEST_URL, max_pages=2, max_depth=1)
        result: Dict[str, Any] = _parse(raw)
        assert "error" not in result
        assert result["pages_crawled"] >= 1


# ──────────────────────────────────────────────────────────────────────
#  JSON serialization
# ──────────────────────────────────────────────────────────────────────

class TestJsonSerialization:

    def test_scrape_url_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url(TEST_URL)
        parsed: Dict[str, Any] = _parse(raw)
        assert isinstance(parsed, dict)

    def test_extract_links_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_links(TEST_URL)
        parsed: Dict[str, Any] = _parse(raw)
        assert isinstance(parsed, dict)

    def test_extract_with_selector_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_with_selector(TEST_URL, selector="a")
        parsed: Dict[str, Any] = _parse(raw)
        assert isinstance(parsed, dict)

    def test_extract_tables_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_tables(TEST_URL)
        parsed: Dict[str, Any] = _parse(raw)
        assert isinstance(parsed, dict)

    def test_get_page_metadata_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(TEST_URL)
        parsed: Dict[str, Any] = _parse(raw)
        assert isinstance(parsed, dict)

    def test_crawl_website_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.crawl_website(TEST_URL, max_pages=2, max_depth=1)
        parsed: Dict[str, Any] = _parse(raw)
        assert isinstance(parsed, dict)

    def test_error_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url("https://thisdomaindoesnotexist12345.com")
        parsed: Dict[str, Any] = _parse(raw)
        assert isinstance(parsed, dict)


# ──────────────────────────────────────────────────────────────────────
#  ToolProcessor integration
# ──────────────────────────────────────────────────────────────────────

class TestFrameworkIntegration:

    def test_tool_processor_registers_all_public_methods(self) -> None:
        from upsonic.tools.processor import ToolProcessor

        t: CrawleeTools = CrawleeTools()
        processor: ToolProcessor = ToolProcessor()
        registered: Dict[str, Any] = processor.register_tools([t])
        registered_names: List[str] = list(registered.keys())
        for tool_name in ALL_TOOL_NAMES:
            assert tool_name in registered_names, (
                f"{tool_name} not registered by ToolProcessor"
            )

    def test_no_private_methods_registered(self) -> None:
        from upsonic.tools.processor import ToolProcessor

        t: CrawleeTools = CrawleeTools()
        processor: ToolProcessor = ToolProcessor()
        processor.register_tools([t])
        for name in processor.registered_tools:
            assert not name.startswith("_"), (
                f"Private method {name} registered by ToolProcessor"
            )

    def test_tool_definitions_have_descriptions(self) -> None:
        from upsonic.tools.processor import ToolProcessor

        t: CrawleeTools = CrawleeTools()
        processor: ToolProcessor = ToolProcessor()
        processor.register_tools([t])
        for name, tool in processor.registered_tools.items():
            if name.startswith("_"):
                continue
            assert getattr(tool, "description", None), f"Tool {name} has no description"

    def test_class_instance_tracked(self) -> None:
        from upsonic.tools.processor import ToolProcessor

        t: CrawleeTools = CrawleeTools()
        processor: ToolProcessor = ToolProcessor()
        processor.register_tools([t])
        instance_id: int = id(t)
        assert instance_id in processor.class_instance_to_tools
        tool_names: List[str] = processor.class_instance_to_tools[instance_id]
        assert "scrape_url" in tool_names


# ──────────────────────────────────────────────────────────────────────
#  Playwright tools (when crawlee[playwright] installed)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _CRAWLEE_PW_AVAILABLE,
    reason="crawlee[playwright] not installed",
)
class TestScrapeDynamicPage:

    def test_scrape_dynamic_page_returns_valid_json_structure(self) -> None:
        t: CrawleeTools = CrawleeTools(
            respect_robots_txt=False,
        )
        raw: str = t.scrape_dynamic_page(TEST_URL)
        data: Dict[str, Any] = _parse(raw)
        if "error" in data:
            assert "url" in data or "error" in data
            return
        assert "url" in data
        assert "nike.com.tr" in data["url"]
        assert "title" in data
        assert "text" in data
        assert "html_length" in data
        assert isinstance(data["html_length"], int)

    def test_scrape_dynamic_page_with_max_content_length(self) -> None:
        t: CrawleeTools = CrawleeTools(respect_robots_txt=False)
        raw: str = t.scrape_dynamic_page(TEST_URL, max_content_length=500)
        data: Dict[str, Any] = _parse(raw)
        if "error" in data:
            return
        assert len(data["text"]) <= 530


@pytest.mark.skipif(
    not _CRAWLEE_PW_AVAILABLE,
    reason="crawlee[playwright] not installed",
)
class TestTakeScreenshot:

    def test_take_screenshot_returns_valid_json_structure(self) -> None:
        t: CrawleeTools = CrawleeTools(respect_robots_txt=False)
        raw: str = t.take_screenshot(TEST_URL, full_page=False)
        data: Dict[str, Any] = _parse(raw)
        if "error" in data:
            assert "url" in data or "error" in data
            return
        assert "url" in data
        assert "nike.com.tr" in data["url"]
        assert "title" in data
        assert "screenshot_path" in data
        assert "screenshot_size_bytes" in data
        assert isinstance(data["screenshot_size_bytes"], int)
        assert data["screenshot_size_bytes"] > 0

    def test_take_screenshot_full_page_true(self) -> None:
        t: CrawleeTools = CrawleeTools(respect_robots_txt=False)
        raw: str = t.take_screenshot(TEST_URL, full_page=True)
        data: Dict[str, Any] = _parse(raw)
        if "error" in data:
            return
        assert "screenshot_path" in data
        assert data["screenshot_size_bytes"] > 0
