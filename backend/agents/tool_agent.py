from typing import Any, Optional

from backend.tools.query_university import query_university
from backend.tools.query_industry import query_industry


_TOOL_REGISTRY = {
    "query_university": query_university,
    "query_industry": query_industry,
}


class ToolAgent:
    """Agent that executes registered tools by name."""

    async def execute(self, tool_name: str, params: dict) -> dict:
        """Execute a tool by name with the given parameters.

        Args:
            tool_name: Name of the tool to execute.
            params: Keyword arguments to pass to the tool.

        Returns:
            dict with keys: success, data, error.
        """
        tool_fn = _TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            return {
                "success": False,
                "data": None,
                "error": f"Unknown tool: {tool_name}",
            }

        try:
            result = tool_fn(**params)
            return {"success": True, "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}
