from mcp.server.fastmcp import FastMCP
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "CryptoService",
    instructions="You are a cryptocurrency assistant that can provide crypto market information.",
    host="0.0.0.0",
    port=8010,
)


@mcp.tool()
async def get_crypto_price(symbol: str, currency: Optional[str] = "USD") -> str:
    """
    Get current cryptocurrency price.

    Args:
        symbol (str): The cryptocurrency symbol (e.g., BTC, ETH)
        currency (str, optional): The currency to display price in. Defaults to "USD".

    Returns:
        str: A string containing mock cryptocurrency price information
    """
    try:
        mock_prices = {"BTC": 45000, "ETH": 2800, "ADA": 0.85, "DOT": 12.50}
        price = mock_prices.get(symbol.upper(), 100.00)
        return f"{symbol.upper()} price (mock): ${price:,.2f} {currency} (24h change: +2.45%)"
    except Exception as e:
        return f"Error getting crypto price: {str(e)}"


@mcp.tool()
async def get_portfolio_value(holdings: str) -> str:
    """
    Calculate portfolio value based on holdings.

    Args:
        holdings (str): Comma-separated list of holdings (e.g., "BTC:0.5,ETH:2.0")

    Returns:
        str: A string containing mock portfolio valuation
    """
    try:
        # Parse holdings format: "BTC:0.5,ETH:2.0"
        total_value = 0
        portfolio_details = []

        for holding in holdings.split(','):
            if ':' in holding:
                symbol, amount = holding.split(':')
                amount = float(amount)
                # Mock calculation
                mock_price = 45000 if symbol.upper() == 'BTC' else 2800
                value = amount * mock_price
                total_value += value
                portfolio_details.append(f"{symbol.upper()}: {amount} @ ${mock_price:,.2f} = ${value:,.2f}")

        return f"Portfolio value (mock): ${total_value:,.2f}\nDetails: {'; '.join(portfolio_details)}"
    except Exception as e:
        return f"Error calculating portfolio value: {str(e)}"


@mcp.tool()
async def get_market_cap(symbol: str) -> str:
    """
    Get market capitalization for a cryptocurrency.

    Args:
        symbol (str): The cryptocurrency symbol

    Returns:
        str: A string containing mock market cap information
    """
    try:
        mock_market_caps = {
            "BTC": "850.5B",
            "ETH": "320.2B",
            "ADA": "28.4B",
            "DOT": "15.7B"
        }
        market_cap = mock_market_caps.get(symbol.upper(), "5.2B")
        return f"{symbol.upper()} market cap (mock): ${market_cap} (Rank: #3 by market cap)"
    except Exception as e:
        return f"Error getting market cap: {str(e)}"


@mcp.tool()
async def get_trending_coins(limit: Optional[int] = 5) -> str:
    """
    Get trending cryptocurrencies.

    Args:
        limit (int, optional): Number of trending coins to show. Defaults to 5.

    Returns:
        str: A string containing mock trending coins
    """
    try:
        trending = ["Bitcoin (BTC): +5.2%", "Ethereum (ETH): +3.8%", "Cardano (ADA): +7.1%", "Polkadot (DOT): +4.5%",
                    "Solana (SOL): +6.3%"]
        return f"Top {limit} trending coins (mock): {', '.join(trending[:limit])}"
    except Exception as e:
        return f"Error getting trending coins: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")