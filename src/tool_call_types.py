"""Tool call type definitions for A2A DataPart content."""

import json
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool call with name and arguments."""

    tool_name: str = Field(description="The name of the tool to call.")
    arguments: dict = Field(description="The arguments to pass to the tool.")

    def __str__(self) -> str:
        return f"ToolCall(tool_name={self.tool_name}, arguments={json.dumps(self.arguments)})"


class ToolCallsData(BaseModel):
    """Data structure for tool calls, embedded in A2A DataPart."""

    tool_calls: list[ToolCall] = Field(
        description="List of tool calls to execute."
    )

    def __str__(self) -> str:
        calls_str = ", ".join(str(tc) for tc in self.tool_calls)
        return f"ToolCallsData([{calls_str}])"
