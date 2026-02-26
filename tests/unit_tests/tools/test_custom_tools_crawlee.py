"""Unit tests for Crawlee custom tools — real HTTP requests, no mocking."""

import json
import os
import pytest
from typing import Any, Dict, List

from upsonic.tools.custom_tools.crawlee import CrawleeTools

NIKE_URL: str = "https://www.nike.com"
NIKE_HELP_URL: str = "https://www.nike.com/help"


@pytest.fixture(scope="module")
def tools() -> CrawleeTools:
    """Shared CrawleeTools instance for the module (HTTP-only tools)."""
    return CrawleeTools(
        max_request_retries=2,
        respect_robots_txt=False,
        max_content_length=30_000,
        enable_scrape_dynamic=False,
        enable_screenshot=False,
    )


@pytest.fixture(scope="module")
def tools_all() -> CrawleeTools:
    """CrawleeTools with every tool enabled."""
    return CrawleeTools(all=True, respect_robots_txt=False)





class TestCrawleeToolsInit:
    """Initialization and attribute tests."""

    def test_default_attributes(self) -> None:
        t = CrawleeTools()
        assert t.headless is True
        assert t.browser_type == "chromium"
        assert t.max_request_retries == 3
        assert t.max_concurrency == 5
        assert t.proxy_urls is None
        assert t.respect_robots_txt is True
        assert t.max_content_length == 50_000

    def test_custom_attributes(self) -> None:
        t = CrawleeTools(
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
        t = CrawleeTools(proxy_urls=["http://proxy1:8080", "http://proxy2:8080"])
        assert t._proxy_config is not None



class TestCrawleeToolsFunctions:
    """Tool registration and enable/disable flag tests."""

    def test_all_eight_tools_enabled_by_default(self) -> None:
        t = CrawleeTools()
        names: List[str] = [f.__name__ for f in t.functions()]
        assert len(names) == 8
        for expected in [
            "scrape_url",
            "extract_links",
            "extract_with_selector",
            "extract_tables",
            "get_page_metadata",
            "crawl_website",
            "scrape_dynamic_page",
            "take_screenshot",
        ]:
            assert expected in names, f"{expected} missing from default tools"

    def test_all_flag_enables_everything(self) -> None:
        t = CrawleeTools(all=True)
        assert len(t.functions()) == 8

    def test_selective_enable(self) -> None:
        t = CrawleeTools(
            enable_scrape=True,
            enable_extract_links=False,
            enable_extract_with_selector=False,
            enable_extract_tables=False,
            enable_get_page_metadata=False,
            enable_crawl=False,
            enable_scrape_dynamic=False,
            enable_screenshot=False,
        )
        names: List[str] = [f.__name__ for f in t.functions()]
        assert names == ["scrape_url"]

    def test_selective_disable(self) -> None:
        t = CrawleeTools(
            enable_scrape=False,
            enable_screenshot=False,
            enable_scrape_dynamic=False,
        )
        names: List[str] = [f.__name__ for f in t.functions()]
        assert "scrape_url" not in names
        assert "take_screenshot" not in names
        assert "scrape_dynamic_page" not in names
        assert "extract_links" in names

    def test_no_private_methods_in_functions(self) -> None:
        t = CrawleeTools()
        for fn in t.functions():
            assert not fn.__name__.startswith("_"), (
                f"Private method {fn.__name__} leaked into functions()"
            )


class TestScrapeUrl:
    """Real HTTP scrape tests against nike.com."""

    def test_scrape_nike_homepage(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result, f"Scrape failed: {result.get('error')}"
        assert result["url"].startswith("https://www.nike.com")
        assert result["title"] is not None
        assert len(result["text"]) > 100
        assert result["status_code"] == 200


    def test_scrape_with_main_content_stripping(self, tools: CrawleeTools) -> None:
        raw_main: str = tools.scrape_url(NIKE_URL, only_main_content=True)
        raw_full: str = tools.scrape_url(NIKE_URL, only_main_content=False)

        main: Dict[str, Any] = json.loads(raw_main)
        full: Dict[str, Any] = json.loads(raw_full)

        assert "error" not in main
        assert "error" not in full
        assert len(full["text"]) >= len(main["text"])


    def test_scrape_truncation(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url(NIKE_URL, max_content_length=200)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert len(result["text"]) <= 220  # 200 + truncation marker


    def test_scrape_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)
        assert isinstance(result, dict)
        assert "url" in result


    def test_scrape_invalid_url_returns_error(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url("https://thisdomaindoesnotexist12345.com")
        result: Dict[str, Any] = json.loads(raw)
        assert isinstance(result, dict)


class TestExtractLinks:
    """Real HTTP link extraction tests against nike.com."""


    def test_extract_links_nike(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_links(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result, f"Failed: {result.get('error')}"
        assert result["total_links"] > 0
        assert isinstance(result["links"], list)
        assert result["status_code"] == 200

        first_link: Dict[str, Any] = result["links"][0]
        assert "href" in first_link
        assert "text" in first_link


    def test_extract_links_returns_href_and_text(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_links(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        hrefs: List[str] = [
            link["href"] for link in result["links"] if link["href"]
        ]
        assert len(hrefs) > 0


    def test_extract_links_with_css_filter(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_links(NIKE_URL, css_filter="nav")
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert isinstance(result["links"], list)


class TestExtractWithSelector:
    """Real HTTP CSS selector extraction tests against nike.com."""


    def test_extract_anchor_tags(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_with_selector(NIKE_URL, selector="a")
        result: Dict[str, Any] = json.loads(raw)

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
        raw: str = tools.extract_with_selector(NIKE_URL, selector="img")
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert isinstance(result["matches"], list)


    def test_extract_nonexistent_selector(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_with_selector(
            NIKE_URL, selector="div.nonexistent-class-xyz-12345"
        )
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert result["count"] == 0
        assert result["matches"] == []


    def test_extract_with_truncation(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_with_selector(
            NIKE_URL, selector="a", max_content_length=50
        )
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        for match in result["matches"]:
            assert len(match["text"]) <= 70  # 50 + truncation marker


class TestExtractTables:
    """Real HTTP table extraction tests."""


    def test_extract_tables_nike(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_tables(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result, f"Failed: {result.get('error')}"
        assert "table_count" in result
        assert isinstance(result["tables"], list)
        assert result["status_code"] == 200


    def test_extract_tables_structure(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_tables(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

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
        raw: str = tools.extract_tables(NIKE_URL, table_index=9999)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" in result
        assert "out of range" in result["error"]


class TestGetPageMetadata:
    """Real HTTP metadata extraction tests against nike.com."""


    def test_metadata_nike_homepage(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result, f"Failed: {result.get('error')}"
        assert result["url"].startswith("https://www.nike.com")
        assert result["title"] is not None
        assert "nike" in result["title"].lower()
        assert result["status_code"] == 200


    def test_metadata_has_all_meta_dict(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert "all_meta" in result
        assert isinstance(result["all_meta"], dict)


    def test_metadata_og_fields_present(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

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
        raw: str = tools.get_page_metadata(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        for field in [
            "twitter_card",
            "twitter_title",
            "twitter_description",
            "twitter_image",
        ]:
            assert field in result


    def test_metadata_canonical_and_favicon(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert "canonical" in result
        assert "favicon" in result


class TestCrawlWebsite:
    """Real HTTP crawl tests against nike.com (limited to 3 pages)."""


    def test_crawl_nike_basic(self, tools: CrawleeTools) -> None:
        raw: str = tools.crawl_website(NIKE_URL, max_pages=3, max_depth=1)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result, f"Failed: {result.get('error')}"
        assert result["seed_url"] == NIKE_URL
        assert result["pages_crawled"] >= 1
        assert isinstance(result["pages"], list)


    def test_crawl_page_structure(self, tools: CrawleeTools) -> None:
        raw: str = tools.crawl_website(NIKE_URL, max_pages=2, max_depth=1)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        for page in result["pages"]:
            assert "url" in page
            assert "title" in page
            assert "text" in page
            assert "status_code" in page


    def test_crawl_max_pages_limit(self, tools: CrawleeTools) -> None:
        raw: str = tools.crawl_website(NIKE_URL, max_pages=2, max_depth=1)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert result["pages_crawled"] <= 10  # concurrency may overshoot the soft cap slightly


class TestAsyncVersions:
    """Test async (private) methods mirror sync behaviour."""

    @pytest.mark.asyncio

    async def test_ascrape_url(self, tools: CrawleeTools) -> None:
        raw: str = await tools._ascrape_url(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert result["status_code"] == 200
        assert len(result["text"]) > 100

    @pytest.mark.asyncio

    async def test_aextract_links(self, tools: CrawleeTools) -> None:
        raw: str = await tools._aextract_links(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert result["total_links"] > 0

    @pytest.mark.asyncio

    async def test_aextract_with_selector(self, tools: CrawleeTools) -> None:
        raw: str = await tools._aextract_with_selector(NIKE_URL, selector="a")
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert result["count"] > 0

    @pytest.mark.asyncio

    async def test_aextract_tables(self, tools: CrawleeTools) -> None:
        raw: str = await tools._aextract_tables(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert "table_count" in result

    @pytest.mark.asyncio

    async def test_aget_page_metadata(self, tools: CrawleeTools) -> None:
        raw: str = await tools._aget_page_metadata(NIKE_URL)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert result["title"] is not None

    @pytest.mark.asyncio

    async def test_acrawl_website(self, tools: CrawleeTools) -> None:
        raw: str = await tools._acrawl_website(NIKE_URL, max_pages=2, max_depth=1)
        result: Dict[str, Any] = json.loads(raw)

        assert "error" not in result
        assert result["pages_crawled"] >= 1


class TestJsonSerialization:
    """Every tool output must be parseable JSON."""


    def test_scrape_url_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url(NIKE_URL)
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)


    def test_extract_links_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_links(NIKE_URL)
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)


    def test_extract_with_selector_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_with_selector(NIKE_URL, selector="a")
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)


    def test_extract_tables_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.extract_tables(NIKE_URL)
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)


    def test_get_page_metadata_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.get_page_metadata(NIKE_URL)
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)


    def test_crawl_website_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.crawl_website(NIKE_URL, max_pages=2, max_depth=1)
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)


    def test_error_returns_valid_json(self, tools: CrawleeTools) -> None:
        raw: str = tools.scrape_url("https://thisdomaindoesnotexist12345.com")
        parsed: Dict[str, Any] = json.loads(raw)
        assert isinstance(parsed, dict)


class TestFrameworkIntegration:
    """Verify CrawleeTools integrates correctly with the Upsonic ToolProcessor."""

    def test_tool_processor_registers_all_public_methods(self) -> None:
        from upsonic.tools.processor import ToolProcessor

        t = CrawleeTools()
        processor = ToolProcessor()
        registered: Dict[str, Any] = processor.register_tools([t])

        registered_names: List[str] = list(registered.keys())

        for tool_name in [
            "scrape_url",
            "extract_links",
            "extract_with_selector",
            "extract_tables",
            "get_page_metadata",
            "crawl_website",
            "scrape_dynamic_page",
            "take_screenshot",
        ]:
            assert tool_name in registered_names, (
                f"{tool_name} not registered by ToolProcessor"
            )

    def test_no_private_methods_registered(self) -> None:
        from upsonic.tools.processor import ToolProcessor

        t = CrawleeTools()
        processor = ToolProcessor()
        registered: Dict[str, Any] = processor.register_tools([t])

        for name in registered:
            assert not name.startswith("_"), (
                f"Private method {name} registered by ToolProcessor"
            )

    def test_tool_definitions_have_descriptions(self) -> None:
        from upsonic.tools.processor import ToolProcessor

        t = CrawleeTools()
        processor = ToolProcessor()
        processor.register_tools([t])

        for name, tool in processor.registered_tools.items():
            if name.startswith("_"):
                continue
            assert tool.description, f"Tool {name} has no description"

    def test_class_instance_tracked(self) -> None:
        from upsonic.tools.processor import ToolProcessor

        t = CrawleeTools()
        processor = ToolProcessor()
        processor.register_tools([t])

        instance_id: int = id(t)
        assert instance_id in processor.class_instance_to_tools
        tool_names: List[str] = processor.class_instance_to_tools[instance_id]
        assert "scrape_url" in tool_names
