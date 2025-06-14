from mcp.server.fastmcp import FastMCP
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "WeatherService",
    instructions="You are a weather assistant that can provide weather information for different locations.",
    host="0.0.0.0",
    port=8001,
)


@mcp.tool()
async def get_weather(location: str, units: Optional[str] = "celsius") -> str:
    """
    Get current weather information for the specified location.

    Args:
        location (str): The location to get weather for
        units (str, optional): Temperature units (celsius/fahrenheit). Defaults to "celsius".

    Returns:
        str: A string containing mock weather information
    """
    try:
        # Mock weather data
        temp = "22°C" if units == "celsius" else "72°F"
        return f"Weather in {location}: Sunny, {temp}, Humidity: 65%, Wind: 10 km/h"
    except Exception as e:
        return f"Error getting weather: {str(e)}"


@mcp.tool()
async def get_forecast(location: str, days: Optional[int] = 3) -> str:
    """
    Get weather forecast for the specified location.

    Args:
        location (str): The location to get forecast for
        days (int, optional): Number of days to forecast. Defaults to 3.

    Returns:
        str: A string containing mock forecast information
    """
    try:
        forecast_text = f"{days}-day forecast for {location}:\n"
        for i in range(1, days + 1):
            forecast_text += f"Day {i}: Partly cloudy, High: 25°C, Low: 18°C\n"
        return forecast_text.strip()
    except Exception as e:
        return f"Error getting forecast: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")