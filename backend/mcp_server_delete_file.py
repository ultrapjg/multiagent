from mcp.server.fastmcp import FastMCP
from datetime import datetime
import pytz
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "FileService",  # Name of the MCP server
    instructions="You are a time assistant that can provide the current time for different timezones.",  # Instructions for the LLM on how to use this tool
    host="0.0.0.0",  # Host address (0.0.0.0 allows connections from any IP)
    port=8005,  # Port number for the server
)


@mcp.tool()
async def delete_file(filename: Optional[str] = "None") -> str:
    """
    Get a filename to delete.

    This function deletes a file.

    Args:
        filename (str, optional): A filename to delete".

    Returns:
        str: result
    """
    try:
        return f"{filename} is deleted."
    except Exception as e:
        return f"Error deleting file: {str(e)}"


if __name__ == "__main__":
    # Start the MCP server with stdio transport
    # stdio transport allows the server to communicate with clients
    # through standard input/output streams, making it suitable for
    # local development and testing
    mcp.run(transport="stdio")
