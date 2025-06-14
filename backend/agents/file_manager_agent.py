from mcp.server.fastmcp import FastMCP
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "FileManagerService",
    instructions="You are a file management assistant that can perform file operations.",
    host="0.0.0.0",
    port=8004,
)


@mcp.tool()
async def list_files(directory: Optional[str] = ".") -> str:
    """
    List files in a directory.

    Args:
        directory (str, optional): The directory to list files from. Defaults to ".".

    Returns:
        str: A string containing mock file listing
    """
    try:
        mock_files = ["document1.txt", "image.jpg", "data.csv", "script.py", "README.md"]
        return f"Files in '{directory}': {', '.join(mock_files)} (5 files total)"
    except Exception as e:
        return f"Error listing files: {str(e)}"


@mcp.tool()
async def create_file(filename: str, content: Optional[str] = "") -> str:
    """
    Create a new file with optional content.

    Args:
        filename (str): The name of the file to create
        content (str, optional): The content to write to the file. Defaults to "".

    Returns:
        str: A string confirming mock file creation
    """
    try:
        size = len(content) if content else 0
        return f"File '{filename}' created successfully (mock) with {size} characters"
    except Exception as e:
        return f"Error creating file: {str(e)}"


@mcp.tool()
async def delete_file(filename: str) -> str:
    """
    Delete a file.

    Args:
        filename (str): The name of the file to delete

    Returns:
        str: A string confirming mock file deletion
    """
    try:
        return f"File '{filename}' deleted successfully (mock)"
    except Exception as e:
        return f"Error deleting file: {str(e)}"


@mcp.tool()
async def get_file_info(filename: str) -> str:
    """
    Get information about a file.

    Args:
        filename (str): The name of the file to get info about

    Returns:
        str: A string containing mock file information
    """
    try:
        return f"File info for '{filename}': Size: 1024 bytes, Modified: 2025-06-14 10:30:00, Type: text/plain (mock)"
    except Exception as e:
        return f"Error getting file info: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")