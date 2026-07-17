"""
mcp_servers/reddit_mcp.py
==========================
Optional MCP server for Reddit community sentiment analysis.

PURPOSE:
- Gauge developer sentiment about technologies
- Find discussions about emerging tools
- Understand community adoption signals

This server is OPTIONAL — the system works without it.
If REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are not set,
this server gracefully returns a "not configured" message.

RELEVANT SUBREDDITS:
- r/MachineLearning
- r/learnmachinelearning
- r/LocalLLaMA
- r/artificial
- r/dataengineering
- r/cscareerquestions
- r/cscareerquestionsIN (India-specific)

TOOLS PROVIDED:
1. search_posts — Search for posts about a topic
2. get_tech_sentiment — Analyze community sentiment for a technology
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config import get_settings
from mcp_servers.base import MCPServer, MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)

# Most relevant subreddits for job market / tech discussions
TECH_SUBREDDITS = [
    "MachineLearning",
    "learnmachinelearning",
    "LocalLLaMA",
    "artificial",
    "dataengineering",
    "cscareerquestions",
    "cscareerquestionsIN",
    "Python",
    "learnprogramming",
]


class RedditMCPServer(MCPServer):
    """
    Optional MCP server providing Reddit discussion analysis.

    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env.
    Sign up at: https://www.reddit.com/prefs/apps
    """

    @property
    def name(self) -> str:
        return "reddit"

    @property
    def description(self) -> str:
        return (
            "Reddit MCP Server — analyzes developer community discussions "
            "to gauge technology sentiment and adoption trends."
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Return all Reddit tools."""
        return [
            {
                "name": "search_posts",
                "description": "Search Reddit for posts about a technology or topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'MCP model context protocol')",
                        },
                        "subreddit": {
                            "type": "string",
                            "description": "Subreddit to search in (optional, searches all if empty)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum posts to return (default: 10)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_tech_sentiment",
                "description": (
                    "Analyze community sentiment about a technology by looking at "
                    "recent Reddit discussions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "technology": {
                            "type": "string",
                            "description": "Technology name to analyze (e.g., 'LangGraph')",
                        },
                        "time_filter": {
                            "type": "string",
                            "description": "Time window: 'week', 'month', 'year' (default: 'month')",
                        },
                    },
                    "required": ["technology"],
                },
            },
        ]

    def _get_reddit_client(self):
        """
        Create a Reddit PRAW client.

        Returns:
            PRAW Reddit instance.

        Raises:
            ImportError: If praw is not installed.
            ValueError: If Reddit credentials are not configured.
        """
        settings = get_settings()
        if not settings.has_reddit:
            raise ValueError(
                "Reddit credentials not configured. "
                "Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to your .env file."
            )

        try:
            # pyrefly: ignore [missing-import]
            import praw
        except ImportError as e:
            raise ImportError("praw is required: pip install praw") from e

        return praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT,
        )

    async def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """Route tool calls to handlers."""
        settings = get_settings()
        if not settings.has_reddit:
            return MCPToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error=(
                    "Reddit MCP not configured. "
                    "Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env to enable."
                ),
            )

        handlers = {
            "search_posts": self._search_posts,
            "get_tech_sentiment": self._get_tech_sentiment,
        }

        handler = handlers.get(call.tool_name)
        if not handler:
            return MCPToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error=f"Unknown tool: {call.tool_name}",
            )

        result_data = await handler(**call.arguments)
        return MCPToolResult(tool_name=call.tool_name, result=result_data, success=True)

    async def _search_posts(
        self,
        query: str,
        subreddit: Optional[str] = None,
        limit: int = 10,
    ) -> dict:
        """Search Reddit posts for a query."""
        reddit = self._get_reddit_client()

        if subreddit:
            search_target = reddit.subreddit(subreddit)
        else:
            # Search across tech subreddits
            combined = "+".join(TECH_SUBREDDITS)
            search_target = reddit.subreddit(combined)

        posts = []
        for submission in search_target.search(query, sort="relevance", limit=limit):
            posts.append({
                "title": submission.title,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "url": f"https://reddit.com{submission.permalink}",
                "subreddit": submission.subreddit.display_name,
                "created_utc": submission.created_utc,
                # Include a snippet of the post body
                "text_snippet": (submission.selftext or "")[:200],
            })

        return {
            "query": query,
            "subreddit": subreddit or "multiple tech subreddits",
            "posts": posts,
            "source": "reddit",
        }

    async def _get_tech_sentiment(
        self,
        technology: str,
        time_filter: str = "month",
    ) -> dict:
        """
        Analyze community sentiment about a technology.

        This is a simple heuristic based on:
        - Number of discussions found
        - Average post score (upvotes - downvotes)
        - Number of comments (engagement signal)

        We NEVER claim this is accurate NLP sentiment analysis —
        it's a rough community activity signal.
        """
        reddit = self._get_reddit_client()
        combined_subs = "+".join(TECH_SUBREDDITS)
        sub = reddit.subreddit(combined_subs)

        posts_data = []
        total_score = 0
        total_comments = 0

        for submission in sub.search(
            technology,
            sort="relevance",
            time_filter=time_filter,
            limit=15,
        ):
            total_score += submission.score
            total_comments += submission.num_comments
            posts_data.append({
                "title": submission.title,
                "score": submission.score,
                "comments": submission.num_comments,
            })

        post_count = len(posts_data)
        avg_score = total_score / post_count if post_count > 0 else 0
        avg_comments = total_comments / post_count if post_count > 0 else 0

        return {
            "technology": technology,
            "time_filter": time_filter,
            "discussion_count": post_count,
            "avg_post_score": round(avg_score, 1),
            "avg_comment_count": round(avg_comments, 1),
            "top_posts": posts_data[:5],
            "source": "reddit",
            "interpretation_note": (
                "Discussion count and scores are community activity signals, "
                "not formal sentiment analysis. Higher scores = more community approval."
            ),
        }
