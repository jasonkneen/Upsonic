"""Unit tests for Firecrawl custom tools."""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import os


FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")


@pytest.fixture
def mock_sync_client() -> Mock:
    """Create a mock synchronous Firecrawl client."""
    client = Mock()
    client.scrape = Mock(return_value={
        "markdown": "# Test Page\n\nThis is test content.",
        "metadata": {
            "title": "Test Page",
            "sourceURL": "https://example.com",
            "statusCode": 200,
        },
    })
    client.crawl = Mock(return_value={
        "status": "completed",
        "total": 3,
        "completed": 3,
        "data": [
            {"markdown": "# Page 1", "metadata": {"sourceURL": "https://example.com/1"}},
            {"markdown": "# Page 2", "metadata": {"sourceURL": "https://example.com/2"}},
            {"markdown": "# Page 3", "metadata": {"sourceURL": "https://example.com/3"}},
        ],
    })
    client.start_crawl = Mock(return_value={
        "id": "crawl-job-123",
        "status": "scraping",
    })
    client.get_crawl_status = Mock(return_value={
        "status": "completed",
        "total": 2,
        "completed": 2,
        "data": [
            {"markdown": "# Page 1", "metadata": {"sourceURL": "https://example.com/1"}},
        ],
    })
    client.cancel_crawl = Mock(return_value=True)
    client.map = Mock(return_value={
        "links": [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/contact",
        ],
    })
    client.search = Mock(return_value={
        "web": [
            {
                "url": "https://example.com",
                "title": "Example",
                "description": "An example page",
                "position": 1,
            },
        ],
    })
    client.batch_scrape = Mock(return_value={
        "status": "completed",
        "total": 2,
        "completed": 2,
        "data": [
            {"markdown": "# URL 1", "metadata": {"sourceURL": "https://example.com/1"}},
            {"markdown": "# URL 2", "metadata": {"sourceURL": "https://example.com/2"}},
        ],
    })
    client.start_batch_scrape = Mock(return_value={
        "id": "batch-job-456",
        "status": "scraping",
    })
    client.get_batch_scrape_status = Mock(return_value={
        "status": "completed",
        "total": 2,
        "completed": 2,
        "data": [
            {"markdown": "# URL 1"},
        ],
    })
    client.extract = Mock(return_value={
        "success": True,
        "data": {
            "company_name": "Firecrawl",
            "description": "Web scraping API",
        },
    })
    client.start_extract = Mock(return_value={
        "id": "extract-job-789",
        "status": "processing",
    })
    client.get_extract_status = Mock(return_value={
        "status": "completed",
        "data": {
            "company_name": "Firecrawl",
        },
    })
    return client


@pytest.fixture
def mock_async_client() -> AsyncMock:
    """Create a mock asynchronous Firecrawl client."""
    client = AsyncMock()
    client.scrape = AsyncMock(return_value={
        "markdown": "# Async Test Page\n\nThis is async test content.",
        "metadata": {
            "title": "Async Test Page",
            "sourceURL": "https://example.com",
            "statusCode": 200,
        },
    })
    client.crawl = AsyncMock(return_value={
        "status": "completed",
        "total": 2,
        "completed": 2,
        "data": [
            {"markdown": "# Async Page 1"},
        ],
    })
    client.start_crawl = AsyncMock(return_value={
        "id": "async-crawl-123",
        "status": "scraping",
    })
    client.get_crawl_status = AsyncMock(return_value={
        "status": "completed",
        "total": 1,
        "completed": 1,
        "data": [],
    })
    client.cancel_crawl = AsyncMock(return_value=True)
    client.map = AsyncMock(return_value={
        "links": ["https://example.com/", "https://example.com/about"],
    })
    client.search = AsyncMock(return_value={
        "web": [
            {
                "url": "https://example.com",
                "title": "Async Example",
                "description": "An async example",
                "position": 1,
            },
        ],
    })
    client.batch_scrape = AsyncMock(return_value={
        "status": "completed",
        "total": 1,
        "completed": 1,
        "data": [{"markdown": "# Async URL 1"}],
    })
    client.start_batch_scrape = AsyncMock(return_value={
        "id": "async-batch-456",
        "status": "scraping",
    })
    client.get_batch_scrape_status = AsyncMock(return_value={
        "status": "completed",
        "total": 1,
        "completed": 1,
        "data": [],
    })
    client.extract = AsyncMock(return_value={
        "success": True,
        "data": {"description": "Async extracted"},
    })
    client.start_extract = AsyncMock(return_value={
        "id": "async-extract-789",
        "status": "processing",
    })
    client.get_extract_status = AsyncMock(return_value={
        "status": "completed",
        "data": {"description": "Done"},
    })
    return client


@pytest.fixture
def firecrawl_tools(mock_sync_client: Mock, mock_async_client: AsyncMock):
    """Create a FirecrawlTools instance with mocked clients."""
    with patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True), \
         patch("upsonic.tools.custom_tools.firecrawl.Firecrawl", return_value=mock_sync_client), \
         patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl", return_value=mock_async_client):
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        tools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)
        tools.sync_client = mock_sync_client
        tools.async_client = mock_async_client
        return tools


class TestFirecrawlToolsInit:
    """Test suite for FirecrawlTools initialization."""

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True)
    @patch("upsonic.tools.custom_tools.firecrawl.Firecrawl")
    @patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl")
    def test_init_with_api_key(self, mock_async_cls: Mock, mock_sync_cls: Mock) -> None:
        """Test initialization with explicit API key."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        tools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)

        assert tools.api_key == FIRECRAWL_API_KEY
        mock_sync_cls.assert_called_once_with(api_key=FIRECRAWL_API_KEY)
        mock_async_cls.assert_called_once_with(api_key=FIRECRAWL_API_KEY)

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True)
    @patch("upsonic.tools.custom_tools.firecrawl.Firecrawl")
    @patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl")
    def test_init_with_env_var(self, mock_async_cls: Mock, mock_sync_cls: Mock) -> None:
        """Test initialization with environment variable."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        with patch.dict("os.environ", {"FIRECRAWL_API_KEY": FIRECRAWL_API_KEY}):
            tools = FirecrawlTools()
            assert tools.api_key == FIRECRAWL_API_KEY

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True)
    @patch("upsonic.tools.custom_tools.firecrawl.Firecrawl")
    @patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl")
    def test_init_missing_api_key(self, mock_async_cls: Mock, mock_sync_cls: Mock) -> None:
        """Test initialization fails without API key."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Firecrawl API key is required"):
                FirecrawlTools(api_key="")

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True)
    @patch("upsonic.tools.custom_tools.firecrawl.Firecrawl")
    @patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl")
    def test_init_with_custom_api_url(self, mock_async_cls: Mock, mock_sync_cls: Mock) -> None:
        """Test initialization with custom API URL for self-hosted."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        custom_url = "https://my-firecrawl.example.com"
        tools = FirecrawlTools(api_key=FIRECRAWL_API_KEY, api_url=custom_url)

        assert tools.api_url == custom_url
        mock_sync_cls.assert_called_once_with(api_key=FIRECRAWL_API_KEY, api_url=custom_url)
        mock_async_cls.assert_called_once_with(api_key=FIRECRAWL_API_KEY, api_url=custom_url)

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True)
    @patch("upsonic.tools.custom_tools.firecrawl.Firecrawl")
    @patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl")
    def test_init_default_config(self, mock_async_cls: Mock, mock_sync_cls: Mock) -> None:
        """Test default configuration values."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        tools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)

        assert tools.default_formats == ["markdown"]
        assert tools.default_scrape_limit == 100
        assert tools.default_search_limit == 5
        assert tools.timeout == 120
        assert tools.poll_interval == 2

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True)
    @patch("upsonic.tools.custom_tools.firecrawl.Firecrawl")
    @patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl")
    def test_init_custom_config(self, mock_async_cls: Mock, mock_sync_cls: Mock) -> None:
        """Test custom configuration values."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        tools = FirecrawlTools(
            api_key=FIRECRAWL_API_KEY,
            default_formats=["markdown", "html"],
            default_scrape_limit=50,
            default_search_limit=10,
            timeout=60,
            poll_interval=5,
        )

        assert tools.default_formats == ["markdown", "html"]
        assert tools.default_scrape_limit == 50
        assert tools.default_search_limit == 10
        assert tools.timeout == 60
        assert tools.poll_interval == 5

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", False)
    def test_init_missing_dependency(self) -> None:
        """Test initialization with missing firecrawl-py dependency."""
        with patch("upsonic.utils.printing.import_error") as mock_error:
            from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
            try:
                FirecrawlTools(api_key=FIRECRAWL_API_KEY)
            except (TypeError, AttributeError, ValueError):
                pass
            mock_error.assert_called_once()


class TestFirecrawlToolsFunctions:
    """Test suite for functions() method and tool enablement."""

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True)
    @patch("upsonic.tools.custom_tools.firecrawl.Firecrawl")
    @patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl")
    def test_default_tools_enabled(self, mock_async_cls: Mock, mock_sync_cls: Mock) -> None:
        """Test that default tools are enabled."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        tools = FirecrawlTools(api_key=FIRECRAWL_API_KEY)
        functions = tools.functions()

        assert len(functions) > 0
        method_names = [f.__name__ for f in functions]
        assert "scrape_url" in method_names
        assert "crawl_website" in method_names
        assert "start_crawl" in method_names
        assert "map_website" in method_names
        assert "search_web" in method_names

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True)
    @patch("upsonic.tools.custom_tools.firecrawl.Firecrawl")
    @patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl")
    def test_all_tools_enabled(self, mock_async_cls: Mock, mock_sync_cls: Mock) -> None:
        """Test that all=True enables every tool."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        tools = FirecrawlTools(api_key=FIRECRAWL_API_KEY, all=True)
        functions = tools.functions()

        expected_methods = [
            "scrape_url", "crawl_website", "start_crawl",
            "map_website", "search_web",
            "batch_scrape", "start_batch_scrape",
            "extract_data", "start_extract",
            "get_crawl_status", "cancel_crawl",
            "get_batch_scrape_status", "get_extract_status",
        ]
        method_names = [f.__name__ for f in functions]
        for expected in expected_methods:
            assert expected in method_names, f"{expected} not in enabled functions"

    @patch("upsonic.tools.custom_tools.firecrawl._FIRECRAWL_AVAILABLE", True)
    @patch("upsonic.tools.custom_tools.firecrawl.Firecrawl")
    @patch("upsonic.tools.custom_tools.firecrawl.AsyncFirecrawl")
    def test_selective_tool_enablement(self, mock_async_cls: Mock, mock_sync_cls: Mock) -> None:
        """Test enabling only specific tools."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        tools = FirecrawlTools(
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
        functions = tools.functions()

        method_names = [f.__name__ for f in functions]
        assert "scrape_url" in method_names
        assert "crawl_website" not in method_names
        assert "search_web" not in method_names


class TestScrapeUrl:
    """Test suite for scrape_url and ascrape_url methods."""

    def test_scrape_url_basic(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test basic URL scraping."""
        result = firecrawl_tools.scrape_url("https://example.com")

        parsed = json.loads(result)
        assert "markdown" in parsed
        assert "# Test Page" in parsed["markdown"]
        mock_sync_client.scrape.assert_called_once()

    def test_scrape_url_with_formats(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test scraping with custom formats."""
        firecrawl_tools.scrape_url(
            "https://example.com",
            formats=["markdown", "html"],
        )

        call_kwargs = mock_sync_client.scrape.call_args
        assert "formats" in call_kwargs.kwargs or "formats" in (call_kwargs[1] if len(call_kwargs) > 1 else {})

    def test_scrape_url_with_json_schema(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test scraping with JSON schema for structured extraction."""
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        }
        firecrawl_tools.scrape_url(
            "https://example.com",
            json_schema=schema,
            json_prompt="Extract the page title",
        )

        mock_sync_client.scrape.assert_called_once()
        call_kwargs = mock_sync_client.scrape.call_args[1]
        formats = call_kwargs["formats"]
        json_format_found = any(
            isinstance(f, dict) and f.get("type") == "json"
            for f in formats
        )
        assert json_format_found

    def test_scrape_url_with_all_options(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test scraping with all optional parameters."""
        firecrawl_tools.scrape_url(
            "https://example.com",
            formats=["markdown"],
            only_main_content=True,
            include_tags=["article", "main"],
            exclude_tags=["nav", "footer"],
            wait_for=3000,
            timeout=60000,
            location="US",
            mobile=True,
            skip_tls_verification=False,
            remove_base64_images=True,
        )

        call_kwargs = mock_sync_client.scrape.call_args[1]
        assert call_kwargs["only_main_content"] is True
        assert call_kwargs["include_tags"] == ["article", "main"]
        assert call_kwargs["exclude_tags"] == ["nav", "footer"]
        assert call_kwargs["wait_for"] == 3000
        assert call_kwargs["timeout"] == 60000
        assert call_kwargs["location"] == "US"
        assert call_kwargs["mobile"] is True
        assert call_kwargs["remove_base64_images"] is True

    def test_scrape_url_error_handling(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test scrape error handling returns error JSON."""
        mock_sync_client.scrape.side_effect = Exception("API rate limit exceeded")

        result = firecrawl_tools.scrape_url("https://example.com")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "rate limit" in parsed["error"]

    @pytest.mark.asyncio
    async def test_ascrape_url_basic(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async URL scraping."""
        result = await firecrawl_tools.ascrape_url("https://example.com")

        parsed = json.loads(result)
        assert "markdown" in parsed
        assert "# Async Test Page" in parsed["markdown"]
        mock_async_client.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_ascrape_url_error_handling(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async scrape error handling."""
        mock_async_client.scrape.side_effect = Exception("Async error")

        result = await firecrawl_tools.ascrape_url("https://example.com")

        parsed = json.loads(result)
        assert "error" in parsed


class TestCrawlWebsite:
    """Test suite for crawl_website and acrawl_website methods."""

    def test_crawl_website_basic(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test basic website crawling."""
        result = firecrawl_tools.crawl_website("https://example.com")

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        assert parsed["total"] == 3
        assert len(parsed["data"]) == 3
        mock_sync_client.crawl.assert_called_once()

    def test_crawl_website_with_options(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test crawling with custom options."""
        firecrawl_tools.crawl_website(
            "https://example.com",
            limit=50,
            scrape_formats=["markdown", "html"],
            exclude_paths=["/admin/*"],
            include_paths=["/docs/*"],
            max_depth=3,
            allowed_domains=["example.com"],
            sitemap="only",
            poll_interval=5,
            timeout=300,
        )

        call_kwargs = mock_sync_client.crawl.call_args[1]
        assert call_kwargs["limit"] == 50
        assert call_kwargs["scrape_options"] == {"formats": ["markdown", "html"]}
        assert call_kwargs["exclude_paths"] == ["/admin/*"]
        assert call_kwargs["include_paths"] == ["/docs/*"]
        assert call_kwargs["max_depth"] == 3
        assert call_kwargs["allowed_domains"] == ["example.com"]
        assert call_kwargs["sitemap"] == "only"
        assert call_kwargs["poll_interval"] == 5
        assert call_kwargs["timeout"] == 300

    def test_crawl_website_uses_default_limit(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test that default limit is applied when not specified."""
        firecrawl_tools.crawl_website("https://example.com")

        call_kwargs = mock_sync_client.crawl.call_args[1]
        assert call_kwargs["limit"] == firecrawl_tools.default_scrape_limit

    def test_crawl_website_error_handling(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test crawl error handling."""
        mock_sync_client.crawl.side_effect = Exception("Crawl failed")

        result = firecrawl_tools.crawl_website("https://example.com")

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_acrawl_website_basic(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async website crawling."""
        result = await firecrawl_tools.acrawl_website("https://example.com")

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        mock_async_client.crawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_acrawl_website_error_handling(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async crawl error handling."""
        mock_async_client.crawl.side_effect = Exception("Async crawl failed")

        result = await firecrawl_tools.acrawl_website("https://example.com")

        parsed = json.loads(result)
        assert "error" in parsed


class TestStartCrawl:
    """Test suite for start_crawl and astart_crawl methods."""

    def test_start_crawl_basic(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test starting a non-blocking crawl."""
        result = firecrawl_tools.start_crawl("https://example.com")

        parsed = json.loads(result)
        assert "id" in parsed
        assert parsed["id"] == "crawl-job-123"
        mock_sync_client.start_crawl.assert_called_once()

    def test_start_crawl_with_options(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test starting crawl with options."""
        firecrawl_tools.start_crawl(
            "https://example.com",
            limit=25,
            scrape_formats=["markdown"],
            exclude_paths=["/private/*"],
            max_depth=2,
        )

        call_kwargs = mock_sync_client.start_crawl.call_args[1]
        assert call_kwargs["limit"] == 25
        assert call_kwargs["scrape_options"] == {"formats": ["markdown"]}
        assert call_kwargs["exclude_paths"] == ["/private/*"]
        assert call_kwargs["max_depth"] == 2

    def test_start_crawl_error_handling(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test start_crawl error handling."""
        mock_sync_client.start_crawl.side_effect = Exception("Failed to start")

        result = firecrawl_tools.start_crawl("https://example.com")

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_astart_crawl_basic(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async non-blocking crawl start."""
        result = await firecrawl_tools.astart_crawl("https://example.com")

        parsed = json.loads(result)
        assert "id" in parsed
        assert parsed["id"] == "async-crawl-123"
        mock_async_client.start_crawl.assert_called_once()


class TestCrawlManagement:
    """Test suite for get_crawl_status and cancel_crawl."""

    def test_get_crawl_status(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test checking crawl status."""
        result = firecrawl_tools.get_crawl_status("crawl-job-123")

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        assert parsed["total"] == 2
        mock_sync_client.get_crawl_status.assert_called_once_with("crawl-job-123")

    def test_get_crawl_status_error(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test crawl status error handling."""
        mock_sync_client.get_crawl_status.side_effect = Exception("Job not found")

        result = firecrawl_tools.get_crawl_status("invalid-id")

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_aget_crawl_status(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async crawl status check."""
        result = await firecrawl_tools.aget_crawl_status("async-crawl-123")

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        mock_async_client.get_crawl_status.assert_called_once_with("async-crawl-123")

    def test_cancel_crawl(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test cancelling a crawl job."""
        result = firecrawl_tools.cancel_crawl("crawl-job-123")

        parsed = json.loads(result)
        assert parsed["cancelled"] is True
        mock_sync_client.cancel_crawl.assert_called_once_with("crawl-job-123")

    def test_cancel_crawl_error(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test cancel crawl error handling."""
        mock_sync_client.cancel_crawl.side_effect = Exception("Cannot cancel")

        result = firecrawl_tools.cancel_crawl("invalid-id")

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_acancel_crawl(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async crawl cancellation."""
        result = await firecrawl_tools.acancel_crawl("async-crawl-123")

        parsed = json.loads(result)
        assert parsed["cancelled"] is True
        mock_async_client.cancel_crawl.assert_called_once_with("async-crawl-123")


class TestMapWebsite:
    """Test suite for map_website and amap_website methods."""

    def test_map_website_basic(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test basic website mapping."""
        result = firecrawl_tools.map_website("https://example.com")

        parsed = json.loads(result)
        assert "links" in parsed
        assert len(parsed["links"]) == 3
        mock_sync_client.map.assert_called_once()

    def test_map_website_with_limit(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test mapping with URL limit."""
        firecrawl_tools.map_website("https://example.com", limit=10)

        call_kwargs = mock_sync_client.map.call_args[1]
        assert call_kwargs["limit"] == 10

    def test_map_website_error_handling(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test map error handling."""
        mock_sync_client.map.side_effect = Exception("Map failed")

        result = firecrawl_tools.map_website("https://example.com")

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_amap_website_basic(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async website mapping."""
        result = await firecrawl_tools.amap_website("https://example.com")

        parsed = json.loads(result)
        assert "links" in parsed
        mock_async_client.map.assert_called_once()

    @pytest.mark.asyncio
    async def test_amap_website_with_limit(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async mapping with URL limit."""
        await firecrawl_tools.amap_website("https://example.com", limit=5)

        call_kwargs = mock_async_client.map.call_args[1]
        assert call_kwargs["limit"] == 5


class TestSearchWeb:
    """Test suite for search_web and asearch_web methods."""

    def test_search_web_basic(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test basic web search."""
        result = firecrawl_tools.search_web("test query")

        parsed = json.loads(result)
        assert "web" in parsed
        assert len(parsed["web"]) > 0
        assert parsed["web"][0]["title"] == "Example"
        mock_sync_client.search.assert_called_once()

    def test_search_web_with_limit(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test search with custom limit."""
        firecrawl_tools.search_web("test query", limit=10)

        call_kwargs = mock_sync_client.search.call_args[1]
        assert call_kwargs["limit"] == 10

    def test_search_web_with_scrape_options(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test search with content scraping."""
        scrape_opts: Dict[str, Any] = {"formats": ["markdown", "links"]}
        firecrawl_tools.search_web("test query", scrape_options=scrape_opts)

        call_kwargs = mock_sync_client.search.call_args[1]
        assert call_kwargs["scrape_options"] == scrape_opts

    def test_search_web_with_location(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test search with geo-targeting."""
        firecrawl_tools.search_web("test query", location="Germany")

        call_kwargs = mock_sync_client.search.call_args[1]
        assert call_kwargs["location"] == "Germany"

    def test_search_web_with_time_filter(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test search with time-based filter."""
        firecrawl_tools.search_web("test query", tbs="qdr:w")

        call_kwargs = mock_sync_client.search.call_args[1]
        assert call_kwargs["tbs"] == "qdr:w"

    def test_search_web_with_sources(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test search with specific result sources."""
        firecrawl_tools.search_web("test query", sources=["web", "news"])

        call_kwargs = mock_sync_client.search.call_args[1]
        assert call_kwargs["sources"] == ["web", "news"]

    def test_search_web_with_categories(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test search with category filters."""
        firecrawl_tools.search_web("test query", categories=["github", "research"])

        call_kwargs = mock_sync_client.search.call_args[1]
        assert call_kwargs["categories"] == ["github", "research"]

    def test_search_web_with_all_options(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test search with all options combined."""
        firecrawl_tools.search_web(
            "test query",
            limit=3,
            scrape_options={"formats": ["markdown"]},
            location="US",
            tbs="qdr:d",
            timeout=30000,
            sources=["web"],
            categories=["pdf"],
        )

        call_kwargs = mock_sync_client.search.call_args[1]
        assert call_kwargs["limit"] == 3
        assert call_kwargs["location"] == "US"
        assert call_kwargs["tbs"] == "qdr:d"
        assert call_kwargs["timeout"] == 30000
        assert call_kwargs["sources"] == ["web"]
        assert call_kwargs["categories"] == ["pdf"]

    def test_search_web_error_handling(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test search error handling."""
        mock_sync_client.search.side_effect = Exception("Search failed")

        result = firecrawl_tools.search_web("test query")

        parsed = json.loads(result)
        assert "error" in parsed

    def test_search_web_uses_default_limit(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test that default search limit is applied."""
        firecrawl_tools.search_web("test query")

        call_kwargs = mock_sync_client.search.call_args[1]
        assert call_kwargs["limit"] == firecrawl_tools.default_search_limit

    @pytest.mark.asyncio
    async def test_asearch_web_basic(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async web search."""
        result = await firecrawl_tools.asearch_web("test query")

        parsed = json.loads(result)
        assert "web" in parsed
        mock_async_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_asearch_web_with_options(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async search with options."""
        await firecrawl_tools.asearch_web(
            "test query",
            limit=3,
            location="France",
            tbs="qdr:m",
        )

        call_kwargs = mock_async_client.search.call_args[1]
        assert call_kwargs["limit"] == 3
        assert call_kwargs["location"] == "France"
        assert call_kwargs["tbs"] == "qdr:m"


class TestBatchScrape:
    """Test suite for batch_scrape and start_batch_scrape methods."""

    def test_batch_scrape_basic(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test basic batch scraping."""
        urls: List[str] = ["https://example.com/1", "https://example.com/2"]
        result = firecrawl_tools.batch_scrape(urls)

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        assert parsed["total"] == 2
        mock_sync_client.batch_scrape.assert_called_once()

    def test_batch_scrape_with_formats(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test batch scraping with custom formats."""
        urls: List[str] = ["https://example.com/1"]
        firecrawl_tools.batch_scrape(urls, formats=["markdown", "html"])

        call_args = mock_sync_client.batch_scrape.call_args
        assert call_args[1]["formats"] == ["markdown", "html"]

    def test_batch_scrape_with_timeout(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test batch scraping with custom timeout."""
        urls: List[str] = ["https://example.com/1"]
        firecrawl_tools.batch_scrape(urls, poll_interval=5, timeout=300)

        call_kwargs = mock_sync_client.batch_scrape.call_args[1]
        assert call_kwargs["poll_interval"] == 5
        assert call_kwargs["timeout"] == 300

    def test_batch_scrape_error_handling(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test batch scrape error handling."""
        mock_sync_client.batch_scrape.side_effect = Exception("Batch failed")

        result = firecrawl_tools.batch_scrape(["https://example.com"])

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_abatch_scrape_basic(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async batch scraping."""
        urls: List[str] = ["https://example.com/1"]
        result = await firecrawl_tools.abatch_scrape(urls)

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        mock_async_client.batch_scrape.assert_called_once()

    def test_start_batch_scrape(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test starting a non-blocking batch scrape."""
        urls: List[str] = ["https://example.com/1", "https://example.com/2"]
        result = firecrawl_tools.start_batch_scrape(urls)

        parsed = json.loads(result)
        assert "id" in parsed
        assert parsed["id"] == "batch-job-456"
        mock_sync_client.start_batch_scrape.assert_called_once()

    def test_start_batch_scrape_with_formats(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test starting batch scrape with custom formats."""
        urls: List[str] = ["https://example.com/1"]
        firecrawl_tools.start_batch_scrape(urls, formats=["html"])

        call_kwargs = mock_sync_client.start_batch_scrape.call_args[1]
        assert call_kwargs["formats"] == ["html"]

    def test_start_batch_scrape_error(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test start_batch_scrape error handling."""
        mock_sync_client.start_batch_scrape.side_effect = Exception("Start failed")

        result = firecrawl_tools.start_batch_scrape(["https://example.com"])

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_astart_batch_scrape(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async non-blocking batch scrape start."""
        urls: List[str] = ["https://example.com/1"]
        result = await firecrawl_tools.astart_batch_scrape(urls)

        parsed = json.loads(result)
        assert "id" in parsed
        assert parsed["id"] == "async-batch-456"

    def test_get_batch_scrape_status(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test checking batch scrape status."""
        result = firecrawl_tools.get_batch_scrape_status("batch-job-456")

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        mock_sync_client.get_batch_scrape_status.assert_called_once_with("batch-job-456")

    def test_get_batch_scrape_status_error(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test batch scrape status error handling."""
        mock_sync_client.get_batch_scrape_status.side_effect = Exception("Not found")

        result = firecrawl_tools.get_batch_scrape_status("invalid-id")

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_aget_batch_scrape_status(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async batch scrape status check."""
        result = await firecrawl_tools.aget_batch_scrape_status("async-batch-456")

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        mock_async_client.get_batch_scrape_status.assert_called_once_with("async-batch-456")


class TestExtractData:
    """Test suite for extract_data and start_extract methods."""

    def test_extract_data_with_prompt(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test extraction with a prompt."""
        result = firecrawl_tools.extract_data(
            urls=["https://example.com"],
            prompt="Extract the company description",
        )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert "data" in parsed
        assert parsed["data"]["company_name"] == "Firecrawl"
        mock_sync_client.extract.assert_called_once()

    def test_extract_data_with_schema(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test extraction with a JSON schema."""
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["company_name"],
        }
        firecrawl_tools.extract_data(
            urls=["https://example.com"],
            schema=schema,
        )

        call_kwargs = mock_sync_client.extract.call_args[1]
        assert call_kwargs["schema"] == schema

    def test_extract_data_with_web_search(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test extraction with web search enabled."""
        firecrawl_tools.extract_data(
            urls=["https://example.com"],
            prompt="Extract info",
            enable_web_search=True,
        )

        call_kwargs = mock_sync_client.extract.call_args[1]
        assert call_kwargs["enable_web_search"] is True

    def test_extract_data_with_wildcards(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test extraction with wildcard URLs."""
        firecrawl_tools.extract_data(
            urls=["https://example.com/*"],
            prompt="Extract all page titles",
        )

        call_kwargs = mock_sync_client.extract.call_args[1]
        assert call_kwargs["urls"] == ["https://example.com/*"]

    def test_extract_data_without_urls(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test extraction without URLs (prompt-only mode)."""
        firecrawl_tools.extract_data(
            prompt="Extract Firecrawl's company mission",
        )

        call_kwargs = mock_sync_client.extract.call_args[1]
        assert "urls" not in call_kwargs
        assert call_kwargs["prompt"] == "Extract Firecrawl's company mission"

    def test_extract_data_with_scrape_options(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test extraction with scrape options."""
        scrape_opts: Dict[str, Any] = {"formats": [{"type": "json"}]}
        firecrawl_tools.extract_data(
            urls=["https://example.com"],
            prompt="Extract",
            scrape_options=scrape_opts,
        )

        call_kwargs = mock_sync_client.extract.call_args[1]
        assert call_kwargs["scrape_options"] == scrape_opts

    def test_extract_data_error_handling(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test extract error handling."""
        mock_sync_client.extract.side_effect = Exception("Extract failed")

        result = firecrawl_tools.extract_data(
            urls=["https://example.com"],
            prompt="Extract",
        )

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_aextract_data_basic(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async data extraction."""
        result = await firecrawl_tools.aextract_data(
            urls=["https://example.com"],
            prompt="Extract description",
        )

        parsed = json.loads(result)
        assert parsed["success"] is True
        mock_async_client.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_aextract_data_error(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async extract error handling."""
        mock_async_client.extract.side_effect = Exception("Async extract failed")

        result = await firecrawl_tools.aextract_data(
            urls=["https://example.com"],
            prompt="Extract",
        )

        parsed = json.loads(result)
        assert "error" in parsed


class TestStartExtract:
    """Test suite for start_extract and get_extract_status methods."""

    def test_start_extract(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test starting a non-blocking extraction job."""
        result = firecrawl_tools.start_extract(
            urls=["https://example.com"],
            prompt="Extract company info",
        )

        parsed = json.loads(result)
        assert "id" in parsed
        assert parsed["id"] == "extract-job-789"
        mock_sync_client.start_extract.assert_called_once()

    def test_start_extract_with_schema(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test starting extraction with schema."""
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }
        firecrawl_tools.start_extract(
            urls=["https://example.com"],
            schema=schema,
        )

        call_args = mock_sync_client.start_extract.call_args
        assert call_args[1]["schema"] == schema

    def test_start_extract_with_web_search(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test starting extraction with web search."""
        firecrawl_tools.start_extract(
            urls=["https://example.com"],
            prompt="Extract",
            enable_web_search=True,
        )

        call_kwargs = mock_sync_client.start_extract.call_args[1]
        assert call_kwargs["enable_web_search"] is True

    def test_start_extract_error(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test start_extract error handling."""
        mock_sync_client.start_extract.side_effect = Exception("Start failed")

        result = firecrawl_tools.start_extract(
            urls=["https://example.com"],
            prompt="Extract",
        )

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_astart_extract(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async non-blocking extraction start."""
        result = await firecrawl_tools.astart_extract(
            urls=["https://example.com"],
            prompt="Extract",
        )

        parsed = json.loads(result)
        assert "id" in parsed
        assert parsed["id"] == "async-extract-789"

    def test_get_extract_status(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test checking extraction job status."""
        result = firecrawl_tools.get_extract_status("extract-job-789")

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        assert "data" in parsed
        mock_sync_client.get_extract_status.assert_called_once_with("extract-job-789")

    def test_get_extract_status_error(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test extract status error handling."""
        mock_sync_client.get_extract_status.side_effect = Exception("Not found")

        result = firecrawl_tools.get_extract_status("invalid-id")

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_aget_extract_status(self, firecrawl_tools: Any, mock_async_client: AsyncMock) -> None:
        """Test async extraction status check."""
        result = await firecrawl_tools.aget_extract_status("async-extract-789")

        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        mock_async_client.get_extract_status.assert_called_once_with("async-extract-789")


class TestClassAttributes:
    """Test suite for class-level constants and attributes."""

    def test_supported_formats(self) -> None:
        """Test SUPPORTED_FORMATS constant."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        expected = ["markdown", "html", "rawHtml", "links", "screenshot", "screenshot@fullPage"]
        assert FirecrawlTools.SUPPORTED_FORMATS == expected

    def test_search_sources(self) -> None:
        """Test SEARCH_SOURCES constant."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        assert "web" in FirecrawlTools.SEARCH_SOURCES
        assert "news" in FirecrawlTools.SEARCH_SOURCES
        assert "images" in FirecrawlTools.SEARCH_SOURCES

    def test_search_categories(self) -> None:
        """Test SEARCH_CATEGORIES constant."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        assert "pdf" in FirecrawlTools.SEARCH_CATEGORIES
        assert "research" in FirecrawlTools.SEARCH_CATEGORIES
        assert "github" in FirecrawlTools.SEARCH_CATEGORIES

    def test_time_based_search_values(self) -> None:
        """Test TIME_BASED_SEARCH_VALUES constant."""
        from upsonic.tools.custom_tools.firecrawl import FirecrawlTools
        tbs = FirecrawlTools.TIME_BASED_SEARCH_VALUES
        assert tbs["past_hour"] == "qdr:h"
        assert tbs["past_day"] == "qdr:d"
        assert tbs["past_week"] == "qdr:w"
        assert tbs["past_month"] == "qdr:m"
        assert tbs["past_year"] == "qdr:y"


class TestJsonSerialization:
    """Test suite for JSON output correctness."""

    def test_scrape_returns_valid_json(self, firecrawl_tools: Any) -> None:
        """Test that scrape returns valid JSON."""
        result = firecrawl_tools.scrape_url("https://example.com")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_crawl_returns_valid_json(self, firecrawl_tools: Any) -> None:
        """Test that crawl returns valid JSON."""
        result = firecrawl_tools.crawl_website("https://example.com")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_search_returns_valid_json(self, firecrawl_tools: Any) -> None:
        """Test that search returns valid JSON."""
        result = firecrawl_tools.search_web("test")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_error_returns_valid_json(self, firecrawl_tools: Any, mock_sync_client: Mock) -> None:
        """Test that errors return valid JSON."""
        mock_sync_client.scrape.side_effect = Exception("Test error")
        result = firecrawl_tools.scrape_url("https://example.com")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "error" in parsed

    def test_map_returns_valid_json(self, firecrawl_tools: Any) -> None:
        """Test that map returns valid JSON."""
        result = firecrawl_tools.map_website("https://example.com")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_extract_returns_valid_json(self, firecrawl_tools: Any) -> None:
        """Test that extract returns valid JSON."""
        result = firecrawl_tools.extract_data(
            urls=["https://example.com"],
            prompt="Extract",
        )
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_batch_returns_valid_json(self, firecrawl_tools: Any) -> None:
        """Test that batch scrape returns valid JSON."""
        result = firecrawl_tools.batch_scrape(["https://example.com"])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
