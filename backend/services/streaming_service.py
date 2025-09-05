import logging
from typing import Any, Dict, List, Optional, Callable, Set
from langchain_core.messages import BaseMessage
import json

logger = logging.getLogger(__name__)


class StreamingService:
    """Centralized service for handling model-specific streaming patterns"""
    
    def __init__(self):
        self.seen_tool_calls: Set[str] = set()
        self.node_chunks: Dict[str, Any] = {}
        
    def reset_state(self):
        """Reset streaming state for new conversation"""
        self.seen_tool_calls.clear()
        self.node_chunks.clear()
    
    def detect_model_type(self, chunk_msg: Any) -> str:
        """Detect whether this is OpenAI or Anthropic based on chunk characteristics"""
        # Check for tool calls first - both models can have them
        if hasattr(chunk_msg, 'tool_calls') and chunk_msg.tool_calls:
            # If it has structured content (list), it's likely Anthropic
            if hasattr(chunk_msg, 'content') and isinstance(chunk_msg.content, list):
                return "anthropic"
            else:
                # Otherwise, assume OpenAI (simpler structure)
                return "openai"
        
        # Check content patterns
        if hasattr(chunk_msg, 'content'):
            content = chunk_msg.content
            # Anthropic often has structured content (lists with dicts)
            if isinstance(content, list):
                return "anthropic"
            # OpenAI typically has simple string content
            elif isinstance(content, str):
                return "openai"
        
        return "unknown"
    
    def handle_agent_chunk(
        self, 
        chunk_msg: Any, 
        node_name: str, 
        callback: Optional[Callable] = None,
        model_type: str = "unknown"
    ) -> bool:
        """
        Handle agent node chunks with model-specific logic
        Returns True if chunk was processed, False otherwise
        """
        if not hasattr(chunk_msg, 'content'):
            return False
            
        # Accumulate chunks for proper spacing (both models)
        if node_name not in self.node_chunks:
            self.node_chunks[node_name] = chunk_msg
        else:
            # Use + operator to properly merge chunks (preserves spacing)
            self.node_chunks[node_name] = self.node_chunks[node_name] + chunk_msg
        
        # Stream content if available and callback exists
        if callback and chunk_msg.content:
            try:
                result = callback({"node": node_name, "content": chunk_msg})
                if hasattr(result, "__await__"):
                    # Return coroutine for caller to await
                    return result
            except Exception as e:
                logger.error(f"Error in agent chunk callback: {e}")
        
        # Handle tool calls in agent response
        if hasattr(chunk_msg, 'tool_calls') and chunk_msg.tool_calls and callback:
            for tool_call in chunk_msg.tool_calls:
                self._handle_tool_call(tool_call, callback)
                
        return True
    
    def handle_tool_chunk(
        self, 
        chunk_msg: Any, 
        node_name: str, 
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Handle tool execution chunks consistently
        Returns True if chunk was processed, False otherwise
        """
        if not hasattr(chunk_msg, 'content') or not chunk_msg.content:
            return False
            
        if callback:
            try:
                result = callback({"node": node_name, "content": chunk_msg})
                if hasattr(result, "__await__"):
                    return result
            except Exception as e:
                logger.error(f"Error in tool chunk callback: {e}")
                
        return True
    
    def _handle_tool_call(self, tool_call: Dict[str, Any], callback: Callable):
        """Handle individual tool call with consistent formatting"""
        tool_id = tool_call.get('id')
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        
        if not tool_id or not tool_name:
            return
            
        # Skip if we've already seen this tool call
        if tool_id in self.seen_tool_calls:
            return
            
        self.seen_tool_calls.add(tool_id)
        
        # Skip Complete tool calls
        if tool_name == 'Complete':
            return
            
        try:
            # Send tool call information
            result = callback({
                "node": "tool_args",
                "content": {
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "args": tool_args
                }
            })
            if hasattr(result, "__await__"):
                return result
        except Exception as e:
            logger.error(f"Error sending tool call via callback: {e}")
    
    def normalize_node_name(self, node_name: str, graph_type: str = "simple") -> str:
        """
        Normalize node names across different graph types for consistent handling
        """
        # Map different graph node names to consistent names
        node_mapping = {
            # ReAct graph nodes (from create_react_agent)
            "__start__": "start",
            "agent": "agent", 
            "tools": "tools",
            "__end__": "end",
            # Custom reflection graph nodes
            "reflect": "reflect",
            # Add more mappings as needed
        }
        
        return node_mapping.get(node_name, node_name)
    
    def extract_final_content(self) -> str:
        """Extract final accumulated content for response"""
        # Prefer agent chunks (properly spaced) over other content
        if 'agent' in self.node_chunks and hasattr(self.node_chunks['agent'], 'content'):
            return str(self.node_chunks['agent'].content)
        
        # Fallback to any accumulated content
        for node_name, chunk in self.node_chunks.items():
            if hasattr(chunk, 'content') and chunk.content:
                return str(chunk.content)
                
        return ""
    
    def handle_anthropic_patterns(
        self, 
        chunk_msg: Any, 
        node_name: str, 
        callback: Optional[Callable] = None
    ) -> bool:
        """Handle Anthropic-specific streaming patterns"""
        if not hasattr(chunk_msg, 'content'):
            return False
            
        content = chunk_msg.content
        
        # Handle structured content (list of content blocks)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        # Handle text content
                        text = item.get("text", "")
                        if text and callback:
                            try:
                                # Create a simplified chunk for streaming
                                text_chunk = type('Chunk', (), {'content': text})()
                                result = callback({"node": node_name, "content": text_chunk})
                                if hasattr(result, "__await__"):
                                    return result
                            except Exception as e:
                                logger.error(f"Error streaming Anthropic text: {e}")
                    
                    elif item.get("type") == "tool_use":
                        # Handle tool use blocks
                        tool_call = {
                            'id': item.get('id'),
                            'name': item.get('name'),
                            'args': item.get('input', {})
                        }
                        self._handle_tool_call(tool_call, callback)
                        
        return True
    
    def handle_openai_patterns(
        self, 
        chunk_msg: Any, 
        node_name: str, 
        callback: Optional[Callable] = None
    ) -> bool:
        """Handle OpenAI-specific streaming patterns"""
        processed = False
        
        # Handle text content first
        if hasattr(chunk_msg, 'content') and chunk_msg.content:
            processed = self.handle_agent_chunk(chunk_msg, node_name, callback, "openai")
        
        # Handle tool calls separately (for mixed responses)
        if hasattr(chunk_msg, 'tool_calls') and chunk_msg.tool_calls and callback:
            for tool_call in chunk_msg.tool_calls:
                self._handle_tool_call(tool_call, callback)
            processed = True
            
        return processed
    
    async def process_stream_chunk(
        self,
        chunk_msg: Any,
        node_name: str,
        graph_type: str,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Main entry point for processing stream chunks
        Automatically detects model type and applies appropriate patterns
        """
        normalized_node = self.normalize_node_name(node_name, graph_type)
        
        # Detect model type if not specified
        model_type = self.detect_model_type(chunk_msg)
        
        if normalized_node in ['agent', 'start']:
            if model_type == "anthropic":
                result = self.handle_anthropic_patterns(chunk_msg, node_name, callback)
            elif model_type == "openai":
                result = self.handle_openai_patterns(chunk_msg, node_name, callback)
            else:
                # Fallback to generic handling
                result = self.handle_agent_chunk(chunk_msg, node_name, callback)
            
            # Await if result is a coroutine
            if hasattr(result, "__await__"):
                await result
                
        elif normalized_node == 'tools':
            result = self.handle_tool_chunk(chunk_msg, node_name, callback)
            if hasattr(result, "__await__"):
                await result
                
        elif normalized_node == 'reflect':
            # Handle reflection node (extended graph only)
            if hasattr(chunk_msg, 'content') and chunk_msg.content and callback:
                try:
                    result = callback({"node": node_name, "content": chunk_msg})
                    if hasattr(result, "__await__"):
                        await result
                except Exception as e:
                    logger.error(f"Error in reflection callback: {e}")
        
        return True