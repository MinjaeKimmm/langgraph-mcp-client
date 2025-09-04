from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class ToolConfig(BaseModel):
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    transport: str = Field(default="stdio", description="Transport method: stdio, http, or sse")
    headers: Optional[dict[str, str]] = Field(default=None, description="HTTP headers for HTTP transport")


class ChatMessage(BaseModel):
    role: str = Field(description="Message role: user, assistant, or system")
    content: str = Field(description="Message content")
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str = Field(description="User message to send to agent")
    model: Optional[str] = Field(default="claude-3-7-sonnet-latest", description="Model to use")
    timeout_seconds: Optional[int] = Field(default=120, description="Timeout in seconds")
    recursion_limit: Optional[int] = Field(default=100, description="Recursion limit")
    thread_id: Optional[str] = None
    enabled_tools: Optional[List[str]] = Field(default=None, description="List of enabled tool server names")


class ToolCall(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Optional[str] = None


class ChatResponse(BaseModel):
    response: str = Field(description="Agent response")
    tool_calls: Optional[List[ToolCall]] = None
    model_used: str
    execution_time: float
    thread_id: str


class StreamingChatResponse(BaseModel):
    """For streaming responses"""
    type: str = Field(description="Type: text, tool_call, or complete")
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    is_complete: bool = False


class AgentStatus(BaseModel):
    initialized: bool
    tool_count: int
    model: str
    available_models: List[str]


class ToolInfo(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    server_name: str


class ServerToolInfo(BaseModel):
    name: str
    description: Optional[str] = None
    tools: List[ToolInfo]
    

class GroupedToolsResponse(BaseModel):
    servers: Dict[str, ServerToolInfo]


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, ToolConfig]


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str] = Field(default_factory=dict)