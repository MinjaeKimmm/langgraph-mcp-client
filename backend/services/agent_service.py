import asyncio
import uuid
import logging
from typing import Any, Dict, List, Callable, Optional
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from services.mcp_service import MCPService
from services.graph_service import GraphService

logger = logging.getLogger(__name__)


# Utility functions
def random_uuid() -> str:
    """Generate a random UUID string"""
    return str(uuid.uuid4())


class AgentService:
    def __init__(self, mcp_service: MCPService):
        self.mcp_service = mcp_service
        self.agent: Optional[CompiledStateGraph] = None
        self.current_model: Optional[str] = None
        self.current_graph_type: str = "simple"  # "simple" or "extended"
        self.checkpointer = MemorySaver()
        self.graph_service = GraphService()
        
        # Output token limits for different models
        self.output_token_info = {
            "claude-3-5-sonnet-latest": {"max_tokens": 8192},
            "claude-3-5-haiku-latest": {"max_tokens": 8192},
            "claude-3-7-sonnet-latest": {"max_tokens": 64000},
            "gpt-4o": {"max_tokens": 16000},
            "gpt-4o-mini": {"max_tokens": 16000},
        }

        # System prompt for the agent
        self.system_prompt = """You are an expert AI assistant with access to powerful tools. Use tools strategically to thoroughly address user requests. When you find relevant information, explore it further if needed. Provide comprehensive, well-structured responses."""

    def get_available_models(self) -> List[str]:
        """Get list of available models based on API keys"""
        import os
        
        available_models = []
        
        # Check Anthropic API key
        if os.environ.get("ANTHROPIC_API_KEY"):
            available_models.extend([
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
            ])
        
        # Check OpenAI API key
        if os.environ.get("OPENAI_API_KEY"):
            available_models.extend(["gpt-4o", "gpt-4o-mini"])
        
        return available_models or ["claude-3-7-sonnet-latest"]  # Default fallback

    async def initialize_agent(self, model_name: str = "claude-3-7-sonnet-latest", enabled_tools: Optional[List[str]] = None, graph_type: str = "simple") -> bool:
        """Initialize agent with specified model and available tools"""
        try:
            # Ensure MCP service is initialized
            if not self.mcp_service.is_initialized():
                await self.mcp_service.initialize()
            
            # Get filtered tools if enabled_tools is provided, otherwise get all tools
            if enabled_tools:
                tools = await self.mcp_service.get_filtered_tools(enabled_tools)
            else:
                tools = await self.mcp_service.get_tools()
            
            if not tools:
                return False

            # Initialize the appropriate model
            if model_name in [
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest", 
                "claude-3-5-haiku-latest",
            ]:
                model = ChatAnthropic(
                    model=model_name,
                    temperature=0.1,
                    max_tokens=self.output_token_info[model_name]["max_tokens"],
                )
            else:  # OpenAI models
                model = ChatOpenAI(
                    model=model_name,
                    temperature=0.1,
                    max_tokens=self.output_token_info[model_name]["max_tokens"],
                )

            # Create agent based on graph type using graph_service
            if graph_type == "simple":
                # Traditional ReAct agent - fast and simple
                self.agent = self.graph_service.create_react_graph(
                    model=model,
                    tools=tools,
                    system_prompt=self.system_prompt,
                    checkpointer=self.checkpointer
                )
            else:  # extended
                # Reflection-based agent - dynamic and thorough
                self.agent = self.graph_service.create_reflection_graph(
                    model=model,
                    tools=tools,
                    system_prompt=self.system_prompt
                )
            
            self.current_model = model_name
            self.current_graph_type = graph_type
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            return False

    async def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        timeout_seconds: int = 120,
        recursion_limit: int = 100,
        callback: Optional[Callable] = None,
        enabled_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Send message to agent and get response"""
        # Reinitialize agent with filtered tools if enabled_tools is provided
        if enabled_tools is not None:
            success = await self.initialize_agent(self.current_model or "claude-3-7-sonnet-latest", enabled_tools, self.current_graph_type)
            if not success:
                raise RuntimeError("Failed to initialize agent with filtered tools")
        elif not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize_agent() first.")

        if not thread_id:
            thread_id = random_uuid()

        try:
            config = RunnableConfig(
                recursion_limit=recursion_limit,
                thread_id=thread_id,
            )

            # Prepare input format based on graph type
            if self.current_graph_type == "simple":
                agent_input = {"messages": [HumanMessage(content=message)]}
            else:  # extended
                # Get tool names synchronously to avoid async issues
                tools = await self.mcp_service.get_tools()
                tool_names = [getattr(t, 'name', str(t)) for t in tools]
                agent_input = {
                    "messages": [],
                    "loop_step": 0,
                    "original_request": message,
                    "tools_available": tool_names,
                    "current_progress": "Starting to work on your request...",
                    "is_complete": False,
                    "max_loops": 10
                }
            
            # Use graph_service unified streaming
            logger.info(f"Starting streaming from graph type: {self.current_graph_type}")
            response = await asyncio.wait_for(
                self.graph_service.astream_graph(
                    graph=self.agent,
                    inputs=agent_input,
                    config=config,
                    callback=callback,
                    graph_type=self.current_graph_type
                ),
                timeout=timeout_seconds,
            )
            
            return {
                "response": response,
                "thread_id": thread_id,
                "model_used": self.current_model,
            }
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Request timed out after {timeout_seconds} seconds")
        except Exception as e:
            raise RuntimeError(f"Error during chat: {str(e)}")

    def is_initialized(self) -> bool:
        """Check if agent is properly initialized"""
        return self.agent is not None

    async def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "initialized": self.is_initialized(),
            "tool_count": await self.mcp_service.get_tool_count(),
            "model": self.current_model or "claude-3-7-sonnet-latest",
            "available_models": self.get_available_models(),
        }