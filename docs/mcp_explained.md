# Model Context Protocol (MCP) Explained

## What is MCP?
The Model Context Protocol (MCP) is an open standard introduced by Anthropic that provides a standardized way for AI models to connect to external data sources and tools.

Think of it like a "USB-C cable for AI applications". Just as USB-C allows any device to connect to any computer without needing custom cables, MCP allows any AI model to connect to any data source without needing custom integration code.

## How it works in this project
In our Job Market Intelligence Agent, we use MCP to isolate our tools into specific "servers":

1. **Filesystem MCP Server**: Safely reads uploaded resumes and saves reports.
2. **GitHub MCP Server**: Fetches repository stats to determine tech trends.
3. **Search MCP Server**: Uses Tavily to search the web for tech news and market reports.
4. **Reddit MCP Server**: Analyzes developer sentiment on subreddits.

## Why use MCP?
1. **Security**: We can restrict what the AI can access. For example, the Filesystem MCP server is strictly limited to reading from the `uploads` directory and writing to the `reports` directory.
2. **Reusability**: These MCP servers can be plugged into any other AI agent framework in the future.
3. **Separation of Concerns**: The agents don't need to know *how* to call the GitHub API; they just ask the MCP registry to execute the `get_repo_stats` tool.
