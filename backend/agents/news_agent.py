from mcp.server.fastmcp import FastMCP
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "NewsService",
    instructions="You are a news assistant that can provide news and current events information.",
    host="0.0.0.0",
    port=8011,
)


@mcp.tool()
async def get_latest_news(category: Optional[str] = "general", limit: Optional[int] = 5) -> str:
    """
    Get latest news articles.

    Args:
        category (str, optional): News category (general/business/tech/sports/health). Defaults to "general".
        limit (int, optional): Number of articles to retrieve. Defaults to 5.

    Returns:
        str: A string containing mock news articles
    """
    try:
        mock_articles = [
            "Tech Giant Announces Revolutionary AI Breakthrough",
            "Global Climate Summit Reaches Historic Agreement",
            "New Medical Treatment Shows Promising Results",
            "Stock Markets Hit Record Highs Amid Economic Recovery",
            "Space Mission Successfully Launches to Mars"
        ]
        return f"Latest {category} news (mock): {limit} articles - {', '.join(mock_articles[:limit])}"
    except Exception as e:
        return f"Error getting latest news: {str(e)}"


@mcp.tool()
async def search_news(query: str, days_back: Optional[int] = 7) -> str:
    """
    Search for news articles matching a query.

    Args:
        query (str): The search query
        days_back (int, optional): How many days back to search. Defaults to 7.

    Returns:
        str: A string containing mock search results
    """
    try:
        return f"News search (mock) for '{query}' (last {days_back} days): Found 12 articles including 'AI Revolution in Healthcare', 'Tech Industry Trends', 'Innovation Summit 2025'"
    except Exception as e:
        return f"Error searching news: {str(e)}"


@mcp.tool()
async def get_trending_topics(region: Optional[str] = "global") -> str:
    """
    Get trending topics in news.

    Args:
        region (str, optional): Region to get trends for. Defaults to "global".

    Returns:
        str: A string containing mock trending topics
    """
    try:
        trending = ["Artificial Intelligence", "Climate Change", "Space Exploration", "Cryptocurrency", "Health Technology"]
        return f"Trending topics (mock) in {region}: {', '.join(trending)} - based on 24h news volume"
    except Exception as e:
        return f"Error getting trending topics: {str(e)}"


@mcp.tool()
async def get_news_summary(article_url: str) -> str:
    """
    Get a summary of a news article.

    Args:
        article_url (str): The URL of the article to summarize

    Returns:
        str: A string containing mock article summary
    """
    try:
        return f"Article summary (mock) for {article_url}: This groundbreaking article discusses recent developments in technology and their impact on society. Key points include innovation trends, market implications, and future predictions. (Summary generated from 850-word article)"
    except Exception as e:
        return f"Error getting news summary: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")