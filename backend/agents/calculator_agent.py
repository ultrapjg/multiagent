from mcp.server.fastmcp import FastMCP
from typing import Optional

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "CalculatorService",
    instructions="You are a calculator assistant that can perform basic mathematical operations.",
    host="0.0.0.0",
    port=8002,
)


@mcp.tool()
async def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate

    Returns:
        str: A string containing the mock calculation result
    """
    try:
        # Mock calculation - just return a formatted response
        return f"Calculation: {expression} = 42 (mock result)"
    except Exception as e:
        return f"Error calculating: {str(e)}"


@mcp.tool()
async def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert units from one type to another.

    Args:
        value (float): The value to convert
        from_unit (str): The source unit
        to_unit (str): The target unit

    Returns:
        str: A string containing the mock conversion result
    """
    try:
        mock_result = value * 2.54  # Mock conversion factor
        return f"Unit conversion: {value} {from_unit} = {mock_result:.2f} {to_unit} (mock result)"
    except Exception as e:
        return f"Error converting units: {str(e)}"


@mcp.tool()
async def solve_equation(equation: str) -> str:
    """
    Solve a mathematical equation.

    Args:
        equation (str): The equation to solve

    Returns:
        str: A string containing the mock solution
    """
    try:
        return f"Equation solution: {equation} => x = 3.14 (mock result)"
    except Exception as e:
        return f"Error solving equation: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")