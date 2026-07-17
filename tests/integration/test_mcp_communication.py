import pytest
from mcp_servers.registry import MCPRegistry

@pytest.fixture
def registry():
    return MCPRegistry()

def test_registry_initialization(registry):
    assert registry.get_server("github") is not None
    assert registry.get_server("filesystem") is not None
    assert registry.get_server("search") is not None

def test_find_tool(registry):
    assert registry.find_tool("get_repo_stats") == ("github", "get_repo_stats")
    assert registry.find_tool("github.get_repo_stats") == ("github", "get_repo_stats")
    assert registry.find_tool("nonexistent_tool") is None
