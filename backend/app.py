import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from models import (
    AgentStatus,
    ChatRequest,
    ChatResponse,
    ConfigUpdateRequest,
    GroupedToolsResponse,
    HealthResponse,
    StreamingChatResponse,
    ToolConfig,
    ToolInfo,
)
from pydantic import ValidationError
from services.agent_service import AgentService
from services.config_service import ConfigService
from services.mcp_service import MCPService

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global services
config_service = ConfigService("config.json")
mcp_service = MCPService(config_service)
agent_service = AgentService(mcp_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting MCP Agent Backend...")
    try:
        # Initialize MCP service
        await mcp_service.initialize()
        tool_count = await mcp_service.get_tool_count()
        logger.info(f"MCP Service initialized with {tool_count} tools")
        
        # Initialize agent with default model
        await agent_service.initialize_agent()
        logger.info("Agent Service initialized")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Agent Backend...")
    await mcp_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="LangGraph MCP Agents Backend",
    description="FastAPI backend for LangGraph agents with MCP tool integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        services={
            "mcp": "initialized" if mcp_service.is_initialized() else "not initialized",
            "agent": "initialized" if agent_service.is_initialized() else "not initialized"
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        services={
            "mcp_service": "healthy" if mcp_service.is_initialized() else "unhealthy",
            "agent_service": "healthy" if agent_service.is_initialized() else "unhealthy",
            "tool_count": str(await mcp_service.get_tool_count())
        }
    )


@app.get("/status", response_model=AgentStatus)
async def get_status():
    """Get agent status"""
    status = await agent_service.get_status()
    return AgentStatus(**status)


@app.get("/tools", response_model=GroupedToolsResponse)
async def get_tools():
    """Get available tools grouped by server"""
    return await mcp_service.get_grouped_tools()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send message to agent and get response"""
    try:
        # Initialize agent if needed or model/graph_type changed
        if (not agent_service.is_initialized() or 
            agent_service.current_model != request.model or
            agent_service.current_graph_type != (request.graph_type or "simple")):
            success = await agent_service.initialize_agent(
                request.model, 
                graph_type=request.graph_type or "simple"
            )
            if not success:
                raise HTTPException(status_code=500, detail="Failed to initialize agent")

        start_time = time.time()
        
        # Process chat request 
        result = await agent_service.chat(
            message=request.message,
            thread_id=request.thread_id,
            timeout_seconds=request.timeout_seconds or 120,
            recursion_limit=request.recursion_limit or 100,
            enabled_tools=request.enabled_tools
        )
        
        execution_time = time.time() - start_time
        
        # Extract the actual response content from the LangGraph result
        response_content = ""
        
        # First, try to get collected content from our modified astream_graph
        if "response" in result and result["response"]:
            if isinstance(result["response"], dict) and "collected_content" in result["response"]:
                response_content = result["response"]["collected_content"]
            else:
                # Fallback to extracting from the final message
                content_obj = result["response"].get("content") if isinstance(result["response"], dict) else getattr(result["response"], "content", result["response"])
                
                # Handle BaseMessage content
                if hasattr(content_obj, "content"):
                    response_content = str(content_obj.content)
                elif isinstance(content_obj, str):
                    response_content = content_obj
                elif isinstance(content_obj, list):
                    # Handle list of content items (like text blocks)
                    text_parts = []
                    for item in content_obj:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                        elif hasattr(item, "text"):
                            text_parts.append(item.text)
                        elif isinstance(item, str):
                            text_parts.append(item)
                    response_content = "".join(text_parts)
                else:
                    response_content = str(content_obj)
        
        return ChatResponse(
            response=response_content,
            model_used=result["model_used"],
            execution_time=execution_time,
            thread_id=result["thread_id"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response with real-time SSE"""
    
    async def generate_stream():
        try:
            # Initialize agent if needed
            if (not agent_service.is_initialized() or 
                agent_service.current_model != request.model or
                agent_service.current_graph_type != (request.graph_type or "simple")):
                success = await agent_service.initialize_agent(
                    request.model, 
                    graph_type=request.graph_type or "simple"
                )
                if not success:
                    yield f"data: {json.dumps({'error': 'Failed to initialize agent'})}\n\n"
                    return

            # Create a queue for real-time streaming
            chunk_queue = asyncio.Queue()
            chat_complete = False
            
            # Track state
            collected_tool_calls = []
            seen_tool_calls = set()
            
            def streaming_callback(chunk: Dict[str, Any]):
                """Callback that uses proper LangGraph tool call protocol"""
                node = chunk.get("node", "")
                content = chunk.get("content", "")
                
                if node == "agent":
                    # Handle direct text content from OpenAI models
                    if hasattr(content, 'content') and isinstance(content.content, str) and content.content.strip():
                        # Send plain text content
                        response = StreamingChatResponse(
                            type="text",
                            content=content.content,
                            is_complete=False
                        )
                        try:
                            chunk_queue.put_nowait(response)
                        except asyncio.QueueFull:
                            pass
                        return response
                    
                    # Handle content with mixed text and tool calls  
                    elif hasattr(content, 'content') and content.content:
                        # Handle structured content (list of content blocks)
                        if isinstance(content.content, list):
                            for item in content.content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        text = item.get("text", "")
                                        if text:
                                            response = StreamingChatResponse(
                                                type="text",
                                                content=text,
                                                is_complete=False
                                            )
                                            try:
                                                chunk_queue.put_nowait(response)
                                            except asyncio.QueueFull:
                                                pass
                                            return response
                                    elif item.get("type") == "tool_use":
                                        # Handle tool calls in content blocks (Anthropic format)
                                        tool_name = item.get('name', 'Unknown')
                                        tool_id = item.get('id', f"{tool_name}_{len(collected_tool_calls)}")
                                        
                                        if tool_name != 'Unknown' and tool_id not in seen_tool_calls:
                                            seen_tool_calls.add(tool_id)
                                            
                                            # Skip streaming "Tool: Complete" messages
                                            if tool_name == 'Complete':
                                                continue
                                            tool_info = f"Tool: {tool_name}"
                                            collected_tool_calls.append(tool_info)
                                            
                                            response = StreamingChatResponse(
                                                type="tool",
                                                content=tool_info,
                                                tool_call_id=tool_id,
                                                is_complete=False
                                            )
                                            try:
                                                chunk_queue.put_nowait(response)
                                            except asyncio.QueueFull:
                                                pass
                                            return response
                        # Handle plain string content
                        elif isinstance(content.content, str) and content.content.strip():
                            response = StreamingChatResponse(
                                type="text",
                                content=content.content,
                                is_complete=False
                            )
                            try:
                                chunk_queue.put_nowait(response)
                            except asyncio.QueueFull:
                                pass
                            return response
                    
                    # Handle standard LangGraph tool calls (if not already processed above)
                    elif hasattr(content, 'tool_calls') and content.tool_calls:
                        for tool_call in content.tool_calls:
                            tool_name = tool_call.get('name', 'Unknown')
                            tool_id = tool_call.get('id', f"{tool_name}_{len(collected_tool_calls)}")
                            
                            if tool_name != 'Unknown' and tool_id not in seen_tool_calls:
                                seen_tool_calls.add(tool_id)
                                
                                # Skip streaming "Tool: Complete" messages
                                if tool_name == 'Complete':
                                    continue
                                tool_info = f"Tool: {tool_name}"
                                collected_tool_calls.append(tool_info)
                                
                                response = StreamingChatResponse(
                                    type="tool",
                                    content=tool_info,
                                    tool_call_id=tool_id,
                                    is_complete=False
                                )
                                try:
                                    chunk_queue.put_nowait(response)
                                except asyncio.QueueFull:
                                    pass
                                return response
                
                elif node == "tool_args":
                    # Handle real tool arguments from graph service
                    if isinstance(content, dict):
                        tool_call_id = content.get('tool_call_id')
                        tool_name = content.get('tool_name', 'unknown')
                        args = content.get('args', {})
                        
                        if tool_name != 'unknown' and tool_call_id not in seen_tool_calls:
                            # First send the tool name (like Anthropic does)
                            seen_tool_calls.add(tool_call_id)
                            tool_info = f"Tool: {tool_name}"
                            collected_tool_calls.append(tool_info)
                            
                            response = StreamingChatResponse(
                                type="tool",
                                content=tool_info,
                                tool_call_id=tool_call_id,
                                is_complete=False
                            )
                            try:
                                chunk_queue.put_nowait(response)
                            except asyncio.QueueFull:
                                pass
                        
                        if args:
                            # Then send the tool input
                            args_content = f"Tool Input: {json.dumps(args, indent=2)}"
                            response = StreamingChatResponse(
                                type="tool",
                                content=args_content,
                                tool_call_id=tool_call_id,
                                is_complete=False
                            )
                            try:
                                chunk_queue.put_nowait(response)
                            except asyncio.QueueFull:
                                pass
                            return response
                
                elif node == "tools":
                    # Handle tool execution results from tools node
                    if hasattr(content, 'content') and content.content:
                        # Get tool info
                        tool_name = getattr(content, 'name', 'unknown')
                        result_tool_id = getattr(content, 'tool_call_id', None)
                        
                        tool_result = f"Tool Result: {content.content}"
                        
                        if tool_result not in collected_tool_calls:
                            collected_tool_calls.append(tool_result)
                            response = StreamingChatResponse(
                                type="tool",
                                content=tool_result,
                                tool_call_id=result_tool_id,
                                is_complete=False
                            )
                            try:
                                chunk_queue.put_nowait(response)
                            except asyncio.QueueFull:
                                pass
                            return response
                
                return None
            
            # Start the chat in a background task
            async def run_chat():
                nonlocal chat_complete
                try:
                    await agent_service.chat(
                        message=request.message,
                        thread_id=request.thread_id,
                        timeout_seconds=request.timeout_seconds or 120,
                        recursion_limit=request.recursion_limit or 100,
                        callback=streaming_callback,
                        enabled_tools=request.enabled_tools
                    )
                finally:
                    chat_complete = True
                    # Signal completion
                    completion_response = StreamingChatResponse(
                        type="complete",
                        content="",
                        is_complete=True
                    )
                    try:
                        chunk_queue.put_nowait(completion_response)
                    except asyncio.QueueFull:
                        pass
            
            # Start chat task
            chat_task = asyncio.create_task(run_chat())
            
            # Yield chunks as they become available in real-time
            while not chat_complete or not chunk_queue.empty():
                try:
                    # Wait for a chunk with timeout
                    chunk_response = await asyncio.wait_for(chunk_queue.get(), timeout=0.1)
                    data = json.dumps(chunk_response.model_dump())
                    yield f"data: {data}\n\n"
                    
                    if chunk_response.type == "complete":
                        break
                        
                except asyncio.TimeoutError:
                    # No chunk available, continue waiting
                    continue
                except Exception as e:
                    # Handle any errors
                    error_response = StreamingChatResponse(
                        type="error",
                        content=str(e),
                        is_complete=True
                    )
                    yield f"data: {json.dumps(error_response.model_dump())}\n\n"
                    break
            
            # Ensure chat task is cleaned up
            if not chat_task.done():
                chat_task.cancel()
                try:
                    await chat_task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            error_response = StreamingChatResponse(
                type="error",
                content=str(e),
                is_complete=True
            )
            yield f"data: {json.dumps(error_response.model_dump())}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )


@app.get("/config")
async def get_config():
    """Get current MCP configuration"""
    return mcp_service.get_config()


@app.post("/config")
async def update_config(request: ConfigUpdateRequest):
    """Update MCP configuration"""
    try:
        # Convert request to dict format expected by service
        config_dict = {}
        for tool_name, tool_config in request.config.items():
            config_dict[tool_name] = tool_config.model_dump(exclude_none=True)
        
        success = await mcp_service.update_config(config_dict)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
        
        # Reinitialize agent with new tools
        await agent_service.initialize_agent(agent_service.current_model or "claude-sonnet-4-20250514")
        
        return {"message": "Configuration updated successfully", "config": config_dict}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/tool")
async def add_tool(tool_name: str, tool_config: dict):
    """Add a single tool to configuration with smart config extraction"""
    try:
        # Validate tool name
        if not tool_name or not tool_name.strip():
            raise HTTPException(status_code=400, detail="Tool name cannot be empty")
        
        # Smart extraction of tool configuration from various formats
        extracted_config = config_service.extract_tool_config(tool_config, tool_name)
        
        # Parse and validate using Pydantic
        try:
            validated_config = ToolConfig(**extracted_config)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Invalid configuration format: {str(e)}")
            
        # Convert to dict for service
        config_dict = validated_config.model_dump(exclude_none=True)
        
        # Test the MCP connection before saving to config
        success = await mcp_service.test_and_add_tool(tool_name, config_dict)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to connect to MCP server '{tool_name}'. Configuration not saved.")
        
        # Reinitialize agent with new tools
        await agent_service.initialize_agent(agent_service.current_model or "claude-sonnet-4-20250514")
        
        return {"message": f"Tool {tool_name} connected successfully and saved to configuration"}
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error adding tool {tool_name}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/config/tool/{tool_name}")
async def remove_tool(tool_name: str):
    """Remove a tool from configuration"""
    try:
        success = await mcp_service.remove_tool(tool_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to remove tool {tool_name}")
        
        # Reinitialize agent
        await agent_service.initialize_agent(agent_service.current_model or "claude-sonnet-4-20250514")
        
        return {"message": f"Tool {tool_name} removed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)