import asyncio
import uuid
from typing import Any, Dict, List, Callable, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from services.mcp_service import MCPService


# Utility functions (originally from utils.py)
def random_uuid() -> str:
    """Generate a random UUID string"""
    return str(uuid.uuid4())


async def astream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    callback: Optional[Callable] = None,
    stream_mode: str = "messages",
) -> Dict[str, Any]:
    """
    Execute LangGraph asynchronously with streaming and callback support
    
    Args:
        graph: Compiled LangGraph object to execute
        inputs: Input dictionary for the graph
        config: Execution configuration (optional)
        callback: Callback function for processing chunks
        stream_mode: Streaming mode ("messages" or "updates")
    
    Returns:
        Dict[str, Any]: Final execution result
    """
    config = config or {}
    final_result = {}
    prev_node = ""

    if stream_mode == "messages":
        collected_content = []
        async for chunk_msg, metadata in graph.astream(
            inputs, config, stream_mode=stream_mode
        ):
            curr_node = metadata["langgraph_node"]
            
            # Collect content from assistant messages - check all chunks from agent node
            if curr_node == 'agent' and hasattr(chunk_msg, 'type'):
                chunk_type = getattr(chunk_msg, 'type', 'unknown')
                content_str = str(getattr(chunk_msg, 'content', ''))
                
                # Only process AI message chunks with actual content  
                if chunk_type == 'AIMessageChunk' and content_str and content_str != '[]' and content_str != '':
                    # Try to parse if it looks like a list string
                    if content_str.startswith('[') and content_str.endswith(']'):
                        try:
                            import ast
                            parsed_content = ast.literal_eval(content_str)
                            if isinstance(parsed_content, list):
                                for content_block in parsed_content:
                                    if isinstance(content_block, dict) and content_block.get('type') == 'text':
                                        text_content = content_block.get('text', '')
                                        collected_content.append(text_content)
                        except Exception:
                            # If parsing fails but it's not empty, add as-is
                            if content_str.strip():
                                collected_content.append(content_str)
                    else:
                        # If it's not a list format but has content, add it
                        collected_content.append(content_str)
            
            final_result = {
                "node": curr_node,
                "content": chunk_msg,
                "metadata": metadata,
            }

            # Execute callback if provided
            if callback:
                result = callback({"node": curr_node, "content": chunk_msg})
                if hasattr(result, "__await__"):
                    await result

            prev_node = curr_node
        
        # Add collected content to final result for easy extraction
        if collected_content:
            final_result["collected_content"] = "".join(collected_content)
        else:
            # If no collected content, try to get it from the final message
            if hasattr(final_result.get("content"), "content"):
                final_result["collected_content"] = str(final_result["content"].content)

    elif stream_mode == "updates":
        async for chunk in graph.astream(inputs, config, stream_mode=stream_mode):
            # Handle different return formats
            if isinstance(chunk, tuple) and len(chunk) == 2:
                namespace, node_chunks = chunk
            else:
                namespace = []  # Empty namespace (root graph)
                node_chunks = chunk  # chunk is the node chunk dictionary

            # Process if it's a dictionary
            if isinstance(node_chunks, dict):
                for node_name, node_chunk in node_chunks.items():
                    final_result = {
                        "node": node_name,
                        "content": node_chunk,
                        "namespace": namespace,
                    }

                    # Execute callback if provided
                    if callback:
                        result = callback({"node": node_name, "content": node_chunk})
                        if hasattr(result, "__await__"):
                            await result

                    prev_node = node_name
            else:
                final_result = {"content": node_chunks}
    else:
        raise ValueError(
            f"Invalid stream_mode: {stream_mode}. Must be 'messages' or 'updates'."
        )

    return final_result


class AgentService:
    def __init__(self, mcp_service: MCPService):
        self.mcp_service = mcp_service
        self.agent: Optional[CompiledStateGraph] = None
        self.current_model: Optional[str] = None
        self.checkpointer = MemorySaver()
        
        # Output token limits for different models
        self.output_token_info = {
            "claude-3-5-sonnet-latest": {"max_tokens": 8192},
            "claude-3-5-haiku-latest": {"max_tokens": 8192},
            "claude-3-7-sonnet-latest": {"max_tokens": 64000},
            "gpt-4o": {"max_tokens": 16000},
            "gpt-4o-mini": {"max_tokens": 16000},
        }

        # System prompt for the agent
        self.system_prompt = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.  
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(concise answer to the question)

**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""

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

    async def initialize_agent(self, model_name: str = "claude-3-7-sonnet-latest") -> bool:
        """Initialize agent with specified model and available tools"""
        try:
            # Ensure MCP service is initialized
            if not self.mcp_service.is_initialized():
                await self.mcp_service.initialize()
            
            tools = self.mcp_service.get_tools()
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

            # Create ReAct agent with tools
            self.agent = create_react_agent(
                model,
                tools,
                checkpointer=self.checkpointer,
                prompt=self.system_prompt,
            )
            self.current_model = model_name
            return True
            
        except Exception as e:
            print(f"Error initializing agent: {str(e)}")
            return False

    async def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        timeout_seconds: int = 120,
        recursion_limit: int = 100,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Send message to agent and get response"""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize_agent() first.")

        if not thread_id:
            thread_id = random_uuid()

        try:
            config = RunnableConfig(
                recursion_limit=recursion_limit,
                thread_id=thread_id,
            )

            response = await asyncio.wait_for(
                astream_graph(
                    self.agent,
                    {"messages": [HumanMessage(content=message)]},
                    callback=callback,
                    config=config,
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

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "initialized": self.is_initialized(),
            "tool_count": self.mcp_service.get_tool_count(),
            "model": self.current_model or "claude-3-7-sonnet-latest",
            "available_models": self.get_available_models(),
        }