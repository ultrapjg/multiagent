from mcp.server.fastmcp import FastMCP
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "SystemMonitorService",
    instructions="You are a system monitoring assistant that can provide system resource information.",
    host="0.0.0.0",
    port=8009,
)


@mcp.tool()
async def get_cpu_usage(interval: Optional[int] = 1) -> str:
    """
    Get CPU usage information.

    Args:
        interval (int, optional): Measurement interval in seconds. Defaults to 1.

    Returns:
        str: A string containing mock CPU usage information
    """
    try:
        return f"CPU Usage (mock, {interval}s interval): 23.4% (Core 1: 21%, Core 2: 25%, Core 3: 24%, Core 4: 23%)"
    except Exception as e:
        return f"Error getting CPU usage: {str(e)}"


@mcp.tool()
async def get_memory_usage() -> str:
    """
    Get memory usage information.

    Returns:
        str: A string containing mock memory usage information
    """
    try:
        return "Memory Usage (mock): 6.2 GB / 16.0 GB (38.8% used), Available: 9.8 GB, Swap: 512 MB / 2.0 GB"
    except Exception as e:
        return f"Error getting memory usage: {str(e)}"


@mcp.tool()
async def get_disk_usage(path: Optional[str] = "/") -> str:
    """
    Get disk usage information for a specific path.

    Args:
        path (str, optional): The path to check disk usage for. Defaults to "/".

    Returns:
        str: A string containing mock disk usage information
    """
    try:
        return f"Disk Usage (mock) for '{path}': 245.6 GB / 500.0 GB (49.1% used), Free: 254.4 GB"
    except Exception as e:
        return f"Error getting disk usage: {str(e)}"


@mcp.tool()
async def get_network_stats(interface: Optional[str] = "all") -> str:
    """
    Get network statistics.

    Args:
        interface (str, optional): Network interface to monitor. Defaults to "all".

    Returns:
        str: A string containing mock network statistics
    """
    try:
        return f"Network Stats (mock) for '{interface}': Bytes sent: 1.2 GB, Bytes received: 3.4 GB, Packets sent: 45678, Packets received: 123456"
    except Exception as e:
        return f"Error getting network stats: {str(e)}"


@mcp.tool()
async def get_system_processes(limit: Optional[int] = 10) -> str:
    """
    Get information about running processes.

    Args:
        limit (int, optional): Number of processes to show. Defaults to 10.

    Returns:
        str: A string containing mock process information
    """
    try:
        mock_processes = ["python (PID: 1234, CPU: 2.3%)", "chrome (PID: 5678, CPU: 15.4%)", "vscode (PID: 9012, CPU: 8.1%)"]
        return f"Top {limit} processes (mock): {', '.join(mock_processes[:limit])}"
    except Exception as e:
        return f"Error getting system processes: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")