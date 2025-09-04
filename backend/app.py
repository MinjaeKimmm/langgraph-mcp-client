from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
import time
from typing import Dict, Any, AsyncGenerator
from dotenv import load_dotenv

from models import (
    ChatRequest, ChatResponse, StreamingChatResponse, 
    AgentStatus, ToolInfo, ConfigUpdateRequest, HealthResponse, ToolConfig
)
from services.config_service import ConfigService
from services.mcp_service import MCPService
from services.agent_service import AgentService

# Load environment variables
load_dotenv(override=True)

# Global services
config_service = ConfigService("mcp_servers/config.json")
mcp_service = MCPService(config_service)
agent_service = AgentService(mcp_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("= Starting MCP Agent Backend...")
    try:
        # Initialize MCP service
        await mcp_service.initialize()
        print(f"MCP Service initialized with {mcp_service.get_tool_count()} tools")
        
        # Initialize agent with default model
        await agent_service.initialize_agent()
        print("Agent Service initialized")
        
    except Exception as e:
        print(f"L Error during startup: {str(e)}")
    
    yield
    
    # Shutdown
    print("= Shutting down MCP Agent Backend...")
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
            "tool_count": str(mcp_service.get_tool_count())
        }
    )


@app.get("/status", response_model=AgentStatus)
async def get_status():
    """Get agent status"""
    return AgentStatus(**agent_service.get_status())


@app.get("/tools", response_model=list[ToolInfo])
async def get_tools():
    """Get available tools"""
    return mcp_service.get_tool_info()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send message to agent and get response"""
    try:
        # Initialize agent if needed or model changed
        if not agent_service.is_initialized() or agent_service.current_model != request.model:
            success = await agent_service.initialize_agent(request.model)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to initialize agent")

        start_time = time.time()
        
        # Process chat request
        result = await agent_service.chat(
            message=request.message,
            thread_id=request.thread_id,
            timeout_seconds=request.timeout_seconds,
            recursion_limit=request.recursion_limit
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
    """Stream chat response"""
    
    async def generate_stream():
        try:
            # Initialize agent if needed
            if not agent_service.is_initialized() or agent_service.current_model != request.model:
                success = await agent_service.initialize_agent(request.model)
                if not success:
                    yield f"data: {json.dumps({'error': 'Failed to initialize agent'})}\n\n"
                    return

            collected_response = []
            collected_tool_calls = []
            seen_tool_calls = set()  # Track seen tool calls to avoid duplicates
            
            def streaming_callback(chunk: Dict[str, Any]):
                """Callback for streaming responses"""
                node = chunk.get("node", "")
                content = chunk.get("content", "")
                
                # Handle AI message chunks
                if hasattr(content, 'content'):
                    if isinstance(content.content, list):
                        for item in content.content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text = item.get("text", "")
                                    if text:
                                        collected_response.append(text)
                                        return StreamingChatResponse(
                                            type="text",
                                            content=text,
                                            is_complete=False
                                        )
                                elif item.get("type") == "tool_use":
                                    # Create unique identifier for this tool call
                                    tool_id = item.get('id', f"{item.get('name', 'Unknown')}_{len(collected_tool_calls)}")
                                    
                                    # Only process if we haven't seen this tool call before
                                    if tool_id not in seen_tool_calls:
                                        seen_tool_calls.add(tool_id)
                                        tool_info = f"Tool: {item.get('name', 'Unknown')}"
                                        if 'input' in item:
                                            tool_info += f"\nInput: {json.dumps(item['input'], indent=2)}"
                                        collected_tool_calls.append(tool_info)
                                        return StreamingChatResponse(
                                            type="tool",
                                            content=tool_info,
                                            is_complete=False
                                        )
                    elif isinstance(content.content, str) and content.content:
                        # Check if this looks like a tool result (contains timestamp patterns or specific formats)
                        content_str = content.content.strip()
                        if ('Current time in' in content_str and 'KST' in content_str) or \
                           ('Asia/Seoul' in content_str and ':' in content_str):
                            # This is likely a tool result, treat as tool output
                            tool_result = f"Tool Output: {content_str}"
                            if tool_result not in collected_tool_calls:
                                collected_tool_calls.append(tool_result)
                                return StreamingChatResponse(
                                    type="tool",
                                    content=tool_result,
                                    is_complete=False
                                )
                        else:
                            # Regular LLM text response
                            collected_response.append(content.content)
                            return StreamingChatResponse(
                                type="text", 
                                content=content.content,
                                is_complete=False
                            )
                        
                # Handle tool messages (tool responses) - only once per unique content
                elif node == 'tools' and hasattr(content, 'content'):
                    # Don't add tool results to collected_response (LLM text)
                    tool_result = f"Tool Result: {content.content}"
                    # Only add if this exact result isn't already collected
                    if tool_result not in collected_tool_calls:
                        collected_tool_calls.append(tool_result)
                        return StreamingChatResponse(
                            type="tool",
                            content=tool_result,
                            is_complete=False
                        )
                
                return None

            # Capture streamed content
            streamed_chunks = []
            
            def streaming_callback_wrapper(chunk):
                response = streaming_callback(chunk)
                if response:
                    streamed_chunks.append(response)
                return response
            
            # Process with streaming
            result = await agent_service.chat(
                message=request.message,
                thread_id=request.thread_id,
                timeout_seconds=request.timeout_seconds,
                recursion_limit=request.recursion_limit,
                callback=streaming_callback_wrapper
            )
            
            # Send all captured streaming chunks
            for chunk_response in streamed_chunks:
                yield f"data: {json.dumps(chunk_response.model_dump())}\n\n"
            
            # Don't send final tool calls summary as they were already streamed
            
            # Send final completion signal
            final_response = StreamingChatResponse(
                type="complete",
                content="",  # Don't send content again, it was already streamed
                is_complete=True
            )
            yield f"data: {json.dumps(final_response.model_dump())}\n\n"
            
        except Exception as e:
            error_response = StreamingChatResponse(
                type="error",
                content=str(e),
                is_complete=True
            )
            yield f"data: {json.dumps(error_response.model_dump())}\n\n"
    
    async def send_chunk(chunk: StreamingChatResponse):
        if chunk:
            return f"data: {json.dumps(chunk.model_dump())}\n\n"
        return ""

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
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
        await agent_service.initialize_agent(agent_service.current_model or "claude-3-7-sonnet-latest")
        
        return {"message": "Configuration updated successfully", "config": config_dict}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/tool")
async def add_tool(tool_name: str, tool_config: ToolConfig):
    """Add a single tool to configuration"""
    try:
        success = await mcp_service.add_tool(tool_name, tool_config.model_dump(exclude_none=True))
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to add tool {tool_name}")
        
        # Reinitialize agent
        await agent_service.initialize_agent(agent_service.current_model or "claude-3-7-sonnet-latest")
        
        return {"message": f"Tool {tool_name} added successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/config/tool/{tool_name}")
async def remove_tool(tool_name: str):
    """Remove a tool from configuration"""
    try:
        success = await mcp_service.remove_tool(tool_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to remove tool {tool_name}")
        
        # Reinitialize agent
        await agent_service.initialize_agent(agent_service.current_model or "claude-3-7-sonnet-latest")
        
        return {"message": f"Tool {tool_name} removed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)