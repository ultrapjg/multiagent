from mcp.server.fastmcp import FastMCP
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "TaskManagerService",
    instructions="You are a task management assistant that can help organize and track tasks.",
    host="0.0.0.0",
    port=8008,
)


@mcp.tool()
async def create_task(title: str, description: Optional[str] = "", priority: Optional[str] = "medium",
                      due_date: Optional[str] = None) -> str:
    """
    Create a new task.

    Args:
        title (str): The task title
        description (str, optional): The task description. Defaults to "".
        priority (str, optional): The task priority (low/medium/high). Defaults to "medium".
        due_date (str, optional): The due date for the task. Defaults to None.

    Returns:
        str: A string confirming mock task creation
    """
    try:
        due_info = f", Due: {due_date}" if due_date else ""
        return f"Task created (mock): '{title}' - Priority: {priority}{due_info}, ID: TASK-{hash(title) % 10000}"
    except Exception as e:
        return f"Error creating task: {str(e)}"


@mcp.tool()
async def list_tasks(status: Optional[str] = "all", priority: Optional[str] = None) -> str:
    """
    List tasks with optional filtering.

    Args:
        status (str, optional): Filter by task status (all/pending/completed). Defaults to "all".
        priority (str, optional): Filter by priority level. Defaults to None.

    Returns:
        str: A string containing mock task list
    """
    try:
        filter_info = f" (Status: {status}" + (f", Priority: {priority}" if priority else "") + ")"
        mock_tasks = [
            "Review project proposal (High)",
            "Update documentation (Medium)",
            "Team meeting preparation (Low)"
        ]
        return f"Tasks{filter_info}: {len(mock_tasks)} found - {', '.join(mock_tasks)}"
    except Exception as e:
        return f"Error listing tasks: {str(e)}"


@mcp.tool()
async def update_task(task_id: str, status: Optional[str] = None, priority: Optional[str] = None) -> str:
    """
    Update a task's status or priority.

    Args:
        task_id (str): The ID of the task to update
        status (str, optional): New status for the task. Defaults to None.
        priority (str, optional): New priority for the task. Defaults to None.

    Returns:
        str: A string confirming mock task update
    """
    try:
        updates = []
        if status:
            updates.append(f"status to '{status}'")
        if priority:
            updates.append(f"priority to '{priority}'")

        update_text = " and ".join(updates) if updates else "no changes"
        return f"Task {task_id} updated (mock): {update_text}"
    except Exception as e:
        return f"Error updating task: {str(e)}"


@mcp.tool()
async def delete_task(task_id: str) -> str:
    """
    Delete a task.

    Args:
        task_id (str): The ID of the task to delete

    Returns:
        str: A string confirming mock task deletion
    """
    try:
        return f"Task {task_id} deleted successfully (mock)"
    except Exception as e:
        return f"Error deleting task: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")