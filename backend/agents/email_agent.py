from mcp.server.fastmcp import FastMCP
from typing import Optional, List

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "EmailService",
    instructions="You are an email assistant that can manage email operations.",
    host="0.0.0.0",
    port=8006,
)


@mcp.tool()
async def send_email(to: str, subject: str, body: str, cc: Optional[str] = None) -> str:
    """
    Send an email to recipients.

    Args:
        to (str): The recipient email address
        subject (str): The email subject
        body (str): The email body content
        cc (str, optional): CC recipients. Defaults to None.

    Returns:
        str: A string confirming mock email sending
    """
    try:
        cc_info = f", CC: {cc}" if cc else ""
        return f"Email sent successfully (mock) - To: {to}{cc_info}, Subject: '{subject}', Body length: {len(body)} characters"
    except Exception as e:
        return f"Error sending email: {str(e)}"


@mcp.tool()
async def check_inbox(limit: Optional[int] = 10) -> str:
    """
    Check inbox for new emails.

    Args:
        limit (int, optional): Maximum number of emails to retrieve. Defaults to 10.

    Returns:
        str: A string containing mock inbox information
    """
    try:
        return f"Inbox check (mock): {limit} recent emails found. Latest: 'Meeting Reminder' from manager@company.com (2 hours ago)"
    except Exception as e:
        return f"Error checking inbox: {str(e)}"


@mcp.tool()
async def search_emails(query: str, folder: Optional[str] = "inbox") -> str:
    """
    Search for emails matching a query.

    Args:
        query (str): The search query
        folder (str, optional): The folder to search in. Defaults to "inbox".

    Returns:
        str: A string containing mock search results
    """
    try:
        return f"Email search (mock) in '{folder}' for '{query}': Found 3 matching emails - 'Project Update', 'Weekly Report', 'Team Meeting'"
    except Exception as e:
        return f"Error searching emails: {str(e)}"


@mcp.tool()
async def mark_as_read(email_id: str) -> str:
    """
    Mark an email as read.

    Args:
        email_id (str): The ID of the email to mark as read

    Returns:
        str: A string confirming the mock operation
    """
    try:
        return f"Email {email_id} marked as read (mock)"
    except Exception as e:
        return f"Error marking email as read: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")