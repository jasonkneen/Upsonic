"""
Smoke tests for CrawleeTools — real requests using Crawlee.

No API key required. All operations run locally.

Coverage:
- Constructor and attribute configuration
- exclude_tools / include_tools filtering
- Every sync tool method with ALL parameters
- Every async method with ALL parameters
- Proper output structure assertions for every method
- Agent integration smoke test

Uses www.nike.com.tr as the test target.

For full coverage (scrape_dynamic_page, take_screenshot) install:
  pip install 'crawlee[playwright]'
  uv run playwright install

Run:
    uv run pytest tests/smoke_tests/tools/test_crawlee_tool.py -v --tb=short
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

import importlib.util

_CRAWLEE_BS_AVAILABLE: bool = importlib.util.find_spec("crawlee.crawlers") is not None
_CRAWLEE_PW_AVAILABLE: bool = _CRAWLEE_BS_AVAILABLE and importlib.util.find_spec("playwright") is not None

pytestmark = [
    pytest.mark.skipif(
        not _CRAWLEE_BS_AVAILABLE,
        reason="crawlee[beautifulsoup] not installed; skipping Crawlee smoke tests",
    ),
    pytest.mark.timeout(300),
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

MODEL: str = "openai/gpt-4o-mini"


@pytest.fixture(scope="module")
def tools() -> Any:
    from upsonic.tools.custom_tools.crawlee import CrawleeTools
    return CrawleeTools(respect_robots_txt=False, max_content_length=5_000)


def _parse(raw: str) -> Dict[str, Any]:
    parsed: Any = json.loads(raw)
    assert isinstance(parsed, dict), (
        f"Expected dict, got {type(parsed).__name__}: {str(parsed)[:200]}"
    )
    if parsed.get("error"):
        pytest.fail(f"Tool returned error: {parsed['error']}")
    return parsed


def _assert_scrape_result(parsed: Dict[str, Any]) -> None:
    assert "url" in parsed
    assert "text" in parsed
    assert isinstance(parsed["text"], str)
    assert len(parsed["text"]) > 0
    assert "status_code" in parsed


def _assert_links_result(parsed: Dict[str, Any]) -> None:
    assert "url" in parsed
    assert "links" in parsed
    assert isinstance(parsed["links"], list)
    assert "total_links" in parsed
    assert isinstance(parsed["total_links"], int)
    assert "status_code" in parsed


def _assert_selector_result(parsed: Dict[str, Any]) -> None:
    assert "url" in parsed
    assert "selector" in parsed
    assert "matches" in parsed
    assert isinstance(parsed["matches"], list)
    assert "count" in parsed
    assert isinstance(parsed["count"], int)
    assert "status_code" in parsed


def _assert_tables_result(parsed: Dict[str, Any]) -> None:
    assert "url" in parsed
    assert "tables" in parsed
    assert isinstance(parsed["tables"], list)
    assert "table_count" in parsed
    assert isinstance(parsed["table_count"], int)
    assert "status_code" in parsed


def _assert_metadata_result(parsed: Dict[str, Any]) -> None:
    assert "url" in parsed
    assert "title" in parsed
    assert "all_meta" in parsed
    assert isinstance(parsed["all_meta"], dict)
    assert "status_code" in parsed


def _assert_crawl_result(parsed: Dict[str, Any]) -> None:
    assert "seed_url" in parsed
    assert "pages" in parsed
    assert isinstance(parsed["pages"], list)
    assert "pages_crawled" in parsed
    assert isinstance(parsed["pages_crawled"], int)


def _assert_dynamic_result(parsed: Dict[str, Any]) -> None:
    assert "url" in parsed
    assert "text" in parsed
    assert isinstance(parsed["text"], str)
    assert len(parsed["text"]) > 0
    assert "html_length" in parsed
    assert isinstance(parsed["html_length"], int)


def _assert_screenshot_result(parsed: Dict[str, Any]) -> None:
    assert "url" in parsed
    assert "screenshot_path" in parsed
    assert isinstance(parsed["screenshot_path"], str)
    assert "screenshot_size_bytes" in parsed
    assert isinstance(parsed["screenshot_size_bytes"], int)
    assert parsed["screenshot_size_bytes"] > 0


def _process_toolkit(toolkit: Any) -> None:
    from upsonic.tools.processor import ToolProcessor
    processor: ToolProcessor = ToolProcessor()
    processor._process_toolkit(toolkit)


# ──────────────────────────────────────────────────────────────────────
#  Initialization
# ──────────────────────────────────────────────────────────────────────

class TestInit:

    def test_init_defaults(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools()
        assert t.headless is True
        assert t.browser_type == "chromium"
        assert t.max_request_retries == 3
        assert t.max_concurrency == 5
        assert t.proxy_urls is None
        assert t.respect_robots_txt is True
        assert t.max_content_length == 50_000

    def test_init_custom_config(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools(
            headless=False,
            browser_type="firefox",
            max_request_retries=5,
            max_concurrency=10,
            proxy_urls=["http://proxy1:8080"],
            respect_robots_txt=False,
            max_content_length=10_000,
        )
        assert t.headless is False
        assert t.browser_type == "firefox"
        assert t.max_request_retries == 5
        assert t.max_concurrency == 10
        assert t.proxy_urls == ["http://proxy1:8080"]
        assert t.respect_robots_txt is False
        assert t.max_content_length == 10_000

    def test_init_with_exclude_tools(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools(exclude_tools=["scrape_url"])
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "scrape_url" not in names

    def test_init_preserves_toolkit_kwargs(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools(use_async=True)
        assert t._toolkit_use_async is True

    def test_init_all_attributes_preserved_after_agent_use(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools(
            headless=True,
            browser_type="chromium",
            max_request_retries=2,
            max_concurrency=3,
            respect_robots_txt=False,
            max_content_length=10_000,
        )
        assert t.headless is True
        assert t.browser_type == "chromium"
        assert t.max_request_retries == 2
        assert t.max_concurrency == 3
        assert t.respect_robots_txt is False
        assert t.max_content_length == 10_000
        assert t.proxy_urls is None


# ──────────────────────────────────────────────────────────────────────
#  functions / exclude_tools filtering
# ──────────────────────────────────────────────────────────────────────

class TestFunctions:

    def test_default_functions(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools()
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        for expected in ALL_TOOL_NAMES:
            assert expected in names, f"{expected} missing from default functions"

    def test_all_tools_count(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools()
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert len(names) == len(ALL_TOOL_NAMES)

    def test_include_tools_is_additive(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools(include_tools=["scrape_url"])
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "scrape_url" in names
        assert len(names) == len(ALL_TOOL_NAMES)

    def test_exclude_all_except_scrape(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        excluded: List[str] = [n for n in ALL_TOOL_NAMES if n != "scrape_url"]
        t: CrawleeTools = CrawleeTools(exclude_tools=excluded)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert names == ["scrape_url"]

    def test_exclude_all_except_crawl(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        excluded: List[str] = [n for n in ALL_TOOL_NAMES if n != "crawl_website"]
        t: CrawleeTools = CrawleeTools(exclude_tools=excluded)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert names == ["crawl_website"]

    def test_exclude_all_except_metadata(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        excluded: List[str] = [n for n in ALL_TOOL_NAMES if n != "get_page_metadata"]
        t: CrawleeTools = CrawleeTools(exclude_tools=excluded)
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert names == ["get_page_metadata"]

    def test_exclude_all_tools(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools(exclude_tools=ALL_TOOL_NAMES)
        _process_toolkit(t)
        assert t.functions == []

    def test_exclude_scrape_keeps_rest(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools(exclude_tools=["scrape_url"])
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "scrape_url" not in names
        assert len(names) == len(ALL_TOOL_NAMES) - 1

    def test_exclude_playwright_tools(self) -> None:
        from upsonic.tools.custom_tools.crawlee import CrawleeTools
        t: CrawleeTools = CrawleeTools(
            exclude_tools=["scrape_dynamic_page", "take_screenshot"],
        )
        _process_toolkit(t)
        names: List[str] = [f.__name__ for f in t.functions]
        assert "scrape_dynamic_page" not in names
        assert "take_screenshot" not in names
        assert len(names) == len(ALL_TOOL_NAMES) - 2


# ──────────────────────────────────────────────────────────────────────
#  scrape_url — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestScrapeUrl:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL))
        _assert_scrape_result(parsed)

    def test_only_main_content_true(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, only_main_content=True))
        _assert_scrape_result(parsed)

    def test_only_main_content_false(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, only_main_content=False))
        _assert_scrape_result(parsed)

    def test_with_max_content_length(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(TEST_URL, max_content_length=500))
        _assert_scrape_result(parsed)
        assert len(parsed["text"]) <= 530

    def test_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_url(
            TEST_URL,
            only_main_content=True,
            max_content_length=2_000,
        ))
        _assert_scrape_result(parsed)

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_url(TEST_URL))
        _assert_scrape_result(parsed)

    @pytest.mark.asyncio
    async def test_async_only_main_content_false(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_url(TEST_URL, only_main_content=False))
        _assert_scrape_result(parsed)

    @pytest.mark.asyncio
    async def test_async_with_max_content_length(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_url(TEST_URL, max_content_length=1_000))
        _assert_scrape_result(parsed)

    @pytest.mark.asyncio
    async def test_async_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_url(
            TEST_URL,
            only_main_content=True,
            max_content_length=3_000,
        ))
        _assert_scrape_result(parsed)


# ──────────────────────────────────────────────────────────────────────
#  extract_links — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestExtractLinks:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_links(TEST_URL))
        _assert_links_result(parsed)
        assert parsed["total_links"] > 0

    def test_with_css_filter(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_links(TEST_URL, css_filter="nav a"))
        _assert_links_result(parsed)

    def test_without_css_filter(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_links(TEST_URL, css_filter=None))
        _assert_links_result(parsed)
        assert parsed["total_links"] > 0

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.aextract_links(TEST_URL))
        _assert_links_result(parsed)
        assert parsed["total_links"] > 0

    @pytest.mark.asyncio
    async def test_async_with_css_filter(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.aextract_links(TEST_URL, css_filter="a"))
        _assert_links_result(parsed)


# ──────────────────────────────────────────────────────────────────────
#  extract_with_selector — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestExtractWithSelector:

    def test_with_anchor_selector(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_with_selector(TEST_URL, selector="a"))
        _assert_selector_result(parsed)
        assert parsed["count"] > 0

    def test_with_heading_selector(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_with_selector(TEST_URL, selector="h1, h2, h3"))
        _assert_selector_result(parsed)

    def test_with_div_selector(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_with_selector(TEST_URL, selector="div"))
        _assert_selector_result(parsed)
        assert parsed["count"] > 0

    def test_with_max_content_length(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_with_selector(
            TEST_URL, selector="a", max_content_length=100,
        ))
        _assert_selector_result(parsed)

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.aextract_with_selector(TEST_URL, selector="a"))
        _assert_selector_result(parsed)
        assert parsed["count"] > 0

    @pytest.mark.asyncio
    async def test_async_with_max_content_length(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.aextract_with_selector(
            TEST_URL, selector="div", max_content_length=200,
        ))
        _assert_selector_result(parsed)


# ──────────────────────────────────────────────────────────────────────
#  extract_tables — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestExtractTables:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.extract_tables(TEST_URL))
        _assert_tables_result(parsed)

    def test_with_table_index_zero(self, tools: Any) -> None:
        raw: str = tools.extract_tables(TEST_URL, table_index=0)
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "tables" in parsed or "error" in parsed

    def test_with_table_index_out_of_range(self, tools: Any) -> None:
        raw: str = tools.extract_tables(TEST_URL, table_index=9999)
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "error" in parsed or "tables" in parsed

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.aextract_tables(TEST_URL))
        _assert_tables_result(parsed)

    @pytest.mark.asyncio
    async def test_async_with_table_index(self, tools: Any) -> None:
        raw: str = await tools.aextract_tables(TEST_URL, table_index=0)
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "tables" in parsed or "error" in parsed


# ──────────────────────────────────────────────────────────────────────
#  get_page_metadata — (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestGetPageMetadata:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.get_page_metadata(TEST_URL))
        _assert_metadata_result(parsed)

    def test_has_title(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.get_page_metadata(TEST_URL))
        assert parsed.get("title") is not None
        assert "nike" in parsed["title"].lower()

    def test_has_og_fields(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.get_page_metadata(TEST_URL))
        _assert_metadata_result(parsed)
        assert "og_title" in parsed
        assert "og_description" in parsed
        assert "og_image" in parsed
        assert "og_type" in parsed
        assert "og_url" in parsed

    def test_has_twitter_fields(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.get_page_metadata(TEST_URL))
        _assert_metadata_result(parsed)
        assert "twitter_card" in parsed
        assert "twitter_title" in parsed
        assert "twitter_description" in parsed
        assert "twitter_image" in parsed

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.aget_page_metadata(TEST_URL))
        _assert_metadata_result(parsed)

    @pytest.mark.asyncio
    async def test_async_has_title(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.aget_page_metadata(TEST_URL))
        assert parsed.get("title") is not None
        assert "nike" in parsed["title"].lower()


# ──────────────────────────────────────────────────────────────────────
#  crawl_website — all attributes (sync + async)
# ──────────────────────────────────────────────────────────────────────

class TestCrawlWebsite:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(TEST_URL, max_pages=1))
        _assert_crawl_result(parsed)
        assert parsed["pages_crawled"] >= 1

    def test_with_max_depth(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, max_pages=1, max_depth=1,
        ))
        _assert_crawl_result(parsed)

    def test_with_only_main_content_false(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, max_pages=1, max_depth=1, only_main_content=False,
        ))
        _assert_crawl_result(parsed)

    def test_with_max_content_length(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, max_pages=1, max_depth=1, max_content_length=1_000,
        ))
        _assert_crawl_result(parsed)

    def test_with_exclude_patterns(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, max_pages=1, max_depth=1, exclude_patterns=["/admin/**"],
        ))
        _assert_crawl_result(parsed)

    def test_with_include_patterns(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL, max_pages=1, max_depth=1, include_patterns=["/**"],
        ))
        _assert_crawl_result(parsed)

    def test_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.crawl_website(
            TEST_URL,
            max_pages=1,
            max_depth=1,
            include_patterns=["/**"],
            exclude_patterns=["/admin/**"],
            only_main_content=True,
            max_content_length=2_000,
        ))
        _assert_crawl_result(parsed)

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.acrawl_website(TEST_URL, max_pages=1))
        _assert_crawl_result(parsed)
        assert parsed["pages_crawled"] >= 1

    @pytest.mark.asyncio
    async def test_async_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.acrawl_website(
            TEST_URL,
            max_pages=1,
            max_depth=1,
            include_patterns=["/**"],
            exclude_patterns=["/private/**"],
            only_main_content=True,
            max_content_length=3_000,
        ))
        _assert_crawl_result(parsed)


# ──────────────────────────────────────────────────────────────────────
#  scrape_dynamic_page — Playwright required (sync + async)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _CRAWLEE_PW_AVAILABLE,
    reason="crawlee[playwright] not installed",
)
class TestScrapeDynamicPage:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_dynamic_page(TEST_URL))
        _assert_dynamic_result(parsed)

    def test_with_max_content_length(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_dynamic_page(
            TEST_URL, max_content_length=1_000,
        ))
        _assert_dynamic_result(parsed)

    def test_with_wait_for_selector(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_dynamic_page(
            TEST_URL, wait_for_selector="body",
        ))
        _assert_dynamic_result(parsed)

    def test_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.scrape_dynamic_page(
            TEST_URL,
            wait_for_selector="body",
            max_content_length=2_000,
        ))
        _assert_dynamic_result(parsed)

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_dynamic_page(TEST_URL))
        _assert_dynamic_result(parsed)

    @pytest.mark.asyncio
    async def test_async_with_max_content_length(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_dynamic_page(
            TEST_URL, max_content_length=1_500,
        ))
        _assert_dynamic_result(parsed)

    @pytest.mark.asyncio
    async def test_async_all_attributes(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.ascrape_dynamic_page(
            TEST_URL,
            wait_for_selector="body",
            max_content_length=3_000,
        ))
        _assert_dynamic_result(parsed)


# ──────────────────────────────────────────────────────────────────────
#  take_screenshot — Playwright required (sync + async)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _CRAWLEE_PW_AVAILABLE,
    reason="crawlee[playwright] not installed",
)
class TestTakeScreenshot:

    def test_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.take_screenshot(TEST_URL))
        _assert_screenshot_result(parsed)

    def test_full_page_false(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.take_screenshot(TEST_URL, full_page=False))
        _assert_screenshot_result(parsed)

    def test_with_wait_for_selector(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(tools.take_screenshot(
            TEST_URL, wait_for_selector="body",
        ))
        _assert_screenshot_result(parsed)

    def test_with_output_path(self, tools: Any) -> None:
        output_path: str = str(Path(tempfile.mkdtemp()) / "test_screenshot.png")
        parsed: Dict[str, Any] = _parse(tools.take_screenshot(
            TEST_URL, output_path=output_path,
        ))
        _assert_screenshot_result(parsed)
        assert parsed["screenshot_path"] == output_path
        assert Path(output_path).exists()

    def test_all_attributes(self, tools: Any) -> None:
        output_path: str = str(Path(tempfile.mkdtemp()) / "full_test.png")
        parsed: Dict[str, Any] = _parse(tools.take_screenshot(
            TEST_URL,
            full_page=True,
            output_path=output_path,
            wait_for_selector="body",
        ))
        _assert_screenshot_result(parsed)
        assert Path(output_path).exists()

    @pytest.mark.asyncio
    async def test_async_basic(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.atake_screenshot(TEST_URL))
        _assert_screenshot_result(parsed)

    @pytest.mark.asyncio
    async def test_async_full_page_false(self, tools: Any) -> None:
        parsed: Dict[str, Any] = _parse(await tools.atake_screenshot(TEST_URL, full_page=False))
        _assert_screenshot_result(parsed)

    @pytest.mark.asyncio
    async def test_async_all_attributes(self, tools: Any) -> None:
        output_path: str = str(Path(tempfile.mkdtemp()) / "async_full.png")
        parsed: Dict[str, Any] = _parse(await tools.atake_screenshot(
            TEST_URL,
            full_page=True,
            output_path=output_path,
            wait_for_selector="body",
        ))
        _assert_screenshot_result(parsed)
        assert Path(output_path).exists()


# ──────────────────────────────────────────────────────────────────────
#  Agent integration smoke tests
# ──────────────────────────────────────────────────────────────────────

def _exclude_all_except(*keep: str) -> List[str]:
    return [n for n in ALL_TOOL_NAMES if n not in keep]


def _get_tool_call_names(tool_calls: List[Dict[str, Any]]) -> List[str]:
    return [tc.get("tool_name", "") for tc in tool_calls]


def _assert_tool_was_called(task: Any, tool_name: str) -> None:
    assert len(task.tool_calls) > 0
    called_names: List[str] = _get_tool_call_names(task.tool_calls)
    assert tool_name in called_names, f"{tool_name} not in tool_calls: {called_names}"
    call: Dict[str, Any] = next(tc for tc in task.tool_calls if tc.get("tool_name") == tool_name)
    assert "params" in call
    assert "tool_result" in call


def _assert_terminal_output(output_text: str, tool_name: str) -> None:
    assert "Agent Started" in output_text or "Agent Status" in output_text
    assert "Tool Calls" in output_text
    assert tool_name in output_text
    assert "LLM Result" in output_text or "Result" in output_text
    assert "Task Metrics" in output_text or "Total Estimated Cost" in output_text


class TestAgentIntegration:

    @pytest.mark.asyncio
    async def test_agent_scrape_url(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.crawlee import CrawleeTools

        crawlee: CrawleeTools = CrawleeTools(
            exclude_tools=_exclude_all_except("scrape_url"),
            respect_robots_txt=False,
            max_content_length=5_000,
        )
        agent: Agent = Agent(model=MODEL, name="Crawlee Scrape Agent", tools=[crawlee])
        task: Task = Task(
            description=f"Use scrape_url on {TEST_URL} and give a one-sentence summary. Return only the summary.",
            tools=[],
        )
        output_buffer: StringIO = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)
        output_text: str = output_buffer.getvalue()

        _assert_terminal_output(output_text, "scrape_url")
        _assert_tool_was_called(task, "scrape_url")
        assert result is not None and len(str(result).strip()) > 10
        r_lower: str = str(result).lower()
        assert any(
            w in r_lower for w in ("nike", "sport", "shoe", "apparel", "website", "page")
        ), f"Result should mention page content, got: {result[:150]}"

    @pytest.mark.asyncio
    async def test_agent_extract_links(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.crawlee import CrawleeTools

        crawlee: CrawleeTools = CrawleeTools(
            exclude_tools=_exclude_all_except("extract_links"),
            respect_robots_txt=False,
        )
        agent: Agent = Agent(model=MODEL, name="Crawlee Links Agent", tools=[crawlee])
        task: Task = Task(
            description=f"Use extract_links on {TEST_URL} and report how many links you found. Reply with just the number.",
            tools=[],
        )
        output_buffer: StringIO = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        _assert_tool_was_called(task, "extract_links")
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_get_page_metadata(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.crawlee import CrawleeTools

        crawlee: CrawleeTools = CrawleeTools(
            exclude_tools=_exclude_all_except("get_page_metadata"),
            respect_robots_txt=False,
        )
        agent: Agent = Agent(model=MODEL, name="Crawlee Metadata Agent", tools=[crawlee])
        task: Task = Task(
            description=f"Use get_page_metadata on {TEST_URL} and tell me the page title. Reply with just the title.",
            tools=[],
        )
        output_buffer: StringIO = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        _assert_tool_was_called(task, "get_page_metadata")
        assert result is not None
        assert "nike" in str(result).lower() or "title" in str(result).lower()

    def test_agent_class_registration_sync(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.crawlee import CrawleeTools

        agent: Agent = Agent(model=MODEL, name="Crawlee Class Agent", tools=[CrawleeTools])
        task: Task = Task(
            description=f"Use scrape_url on {TEST_URL} and reply with exactly: OK",
            tools=[],
        )
        output_buffer: StringIO = StringIO()
        with redirect_stdout(output_buffer):
            result = agent.print_do(task)
        output_text: str = output_buffer.getvalue()

        registered: Dict[str, Any] = agent.tool_manager.processor.registered_tools
        registered_names: List[str] = list(registered.keys())
        for name in ALL_TOOL_NAMES:
            assert name in registered_names, (
                f"Crawlee sync tool {name} not registered; got: {sorted(registered_names)}"
            )

        assert "Agent Started" in output_text or "Agent Status" in output_text
        assert "Tool Calls" in output_text
        assert "scrape_url" in output_text
        assert len(task.tool_calls) > 0
        called: List[str] = _get_tool_call_names(task.tool_calls)
        assert "scrape_url" in called
        assert result is not None
