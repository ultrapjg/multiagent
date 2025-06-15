from mcp.server.fastmcp import FastMCP
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "DatabaseService",
    instructions="You are a database assistant that can perform database operations.",
    host="0.0.0.0",
    port=8007,
)


@mcp.tool()
async def execute_query(query: str, database: Optional[str] = "default") -> str:
    """
    Execute a SQL query on the database.

    Args:
        query (str): The SQL query to execute
        database (str, optional): The database name. Defaults to "default".

    Returns:
        str: A string containing mock query results
    """
    try:
        if query.upper().startswith("SELECT"):
            return f"Query executed (mock) on '{database}': Found 15 rows matching your SELECT query"
        elif query.upper().startswith("INSERT"):
            return f"Query executed (mock) on '{database}': 1 row inserted successfully"
        elif query.upper().startswith("UPDATE"):
            return f"Query executed (mock) on '{database}': 3 rows updated successfully"
        elif query.upper().startswith("DELETE"):
            return f"Query executed (mock) on '{database}': 2 rows deleted successfully"
        else:
            return f"Query executed (mock) on '{database}': Operation completed successfully"
    except Exception as e:
        return f"Error executing query: {str(e)}"


@mcp.tool()
async def get_table_info(table_name: str, database: Optional[str] = "default") -> str:
    """
    Get information about a database table.

    Args:
        table_name (str): The name of the table
        database (str, optional): The database name. Defaults to "default".

    Returns:
        str: A string containing mock table information
    """
    try:
        return f"Table info (mock) for '{table_name}' in '{database}': 5 columns (id, name, email, created_at, status), 1247 rows"
    except Exception as e:
        return f"Error getting table info: {str(e)}"


@mcp.tool()
async def backup_database(database: str, backup_name: Optional[str] = None) -> str:
    """
    Create a backup of the database.

    Args:
        database (str): The database to backup
        backup_name (str, optional): The name for the backup. Defaults to None.

    Returns:
        str: A string confirming mock backup creation
    """
    try:
        backup_name = backup_name or f"{database}_backup_{int(__import__('time').time())}"
        return f"Database backup (mock) created: '{backup_name}' for database '{database}' (Size: 45.2 MB)"
    except Exception as e:
        return f"Error creating backup: {str(e)}"


@mcp.tool()
async def get_database_stats(database: Optional[str] = "default") -> str:
    """
    Get statistics about the database.

    Args:
        database (str, optional): The database name. Defaults to "default".

    Returns:
        str: A string containing mock database statistics
    """
    try:
        stats = {
            "tables": 12,
            "total_rows": 45678,
            "size": "128.5 MB",
            "connections": 5
        }
        return f"Database stats (mock) for '{database}': {stats['tables']} tables, {stats['total_rows']} total rows, {stats['size']} size, {stats['connections']} active connections"
    except Exception as e:
        return f"Error getting database stats: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")