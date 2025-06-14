from mcp.server.fastmcp import FastMCP
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "TranslatorService",
    instructions="You are a translation assistant that can translate text between different languages.",
    host="0.0.0.0",
    port=8003,
)


@mcp.tool()
async def translate_text(text: str, target_language: str, source_language: Optional[str] = "auto") -> str:
    """
    Translate text from one language to another.

    Args:
        text (str): The text to translate
        target_language (str): The target language code
        source_language (str, optional): The source language code. Defaults to "auto".

    Returns:
        str: A string containing the mock translation result
    """
    try:
        return f"Translation ({source_language} → {target_language}): '{text}' → 'Mock translated text' (confidence: 95%)"
    except Exception as e:
        return f"Error translating text: {str(e)}"


@mcp.tool()
async def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        str: A string containing the mock language detection result
    """
    try:
        return f"Language detection: '{text}' is detected as Korean (confidence: 89%)"
    except Exception as e:
        return f"Error detecting language: {str(e)}"


@mcp.tool()
async def get_supported_languages() -> str:
    """
    Get list of supported languages for translation.

    Returns:
        str: A string containing mock supported languages
    """
    try:
        languages = ["English (en)", "Korean (ko)", "Japanese (ja)", "Chinese (zh)", "Spanish (es)", "French (fr)"]
        return f"Supported languages: {', '.join(languages)}"
    except Exception as e:
        return f"Error getting supported languages: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")