import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, TypedDict, Sequence, Literal, Callable
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)


class ReflectionState(TypedDict):
    """State for Reflection-based agent"""
    messages: List[BaseMessage]  # Conversation messages
    loop_step: int               # Current iteration count
    original_request: str        # The user's original request
    tools_available: List[str]   # Available MCP tools
    current_progress: str        # What we've accomplished so far
    is_complete: bool           # Whether we're done
    max_loops: int              # Maximum number of loops


class GraphService:
    """Service for creating and managing both ReAct and Reflection graphs with unified streaming"""
    
    def __init__(self):
        self.current_callback = None  # Store callback for tool nodes to access
        self.agent_prompt = """You are helping a user accomplish their request using available tools.

USER REQUEST: {original_request}
PROGRESS SO FAR: {current_progress}
AVAILABLE TOOLS: {tools}

Rules:
- Give direct, concise responses continuing from current progress without "Based on the current progress..." summaries
- Speak directly to the user (use "you" not "the user")  
- When you have sufficient information, provide a clear, organized summary to the user, then call "Complete"
- Be decisive - if you've found relevant information, that's usually enough
- Don't over-explore unless the user specifically asks for comprehensive analysis
- Focus on giving them actionable results efficiently

What should you do next? Use a tool, provide a summary and call "Complete", or just call "Complete" if you're done."""

        self.reflection_prompt = """You are evaluating whether the current work is complete and satisfactory.

ORIGINAL REQUEST: {original_request}
WORK COMPLETED: {work_completed}

Evaluate:
1. **Completeness** - Has the user's request been fully addressed?
2. **Quality** - Is the information thorough and accurate?
3. **Usefulness** - Would this satisfy the user's needs?

If not complete, what specific actions should be taken next? Be specific about which tools to use and what to look for.

Return either:
- "COMPLETE: [summary]" if satisfied
- "CONTINUE: [specific next actions needed]" if more work is needed"""

    def create_reflection_graph(
        self, 
        model: Runnable, 
        tools: Sequence[Any],
        system_prompt: str = ""
    ) -> CompiledStateGraph:
        """Create a Reflection-based agent graph"""
        
        async def agent_node(state: ReflectionState) -> Dict[str, Any]:
            """Main agent node - chooses and executes tools"""
            try:
                # Get tool names for the prompt
                tool_names = [getattr(tool, 'name', str(tool)) for tool in tools]
                tool_names.append("Complete")  # Add our completion tool
                
                # Build the agent prompt
                agent_message = self.agent_prompt.format(
                    tools=", ".join(tool_names),
                    original_request=state["original_request"],
                    current_progress=state["current_progress"]
                )
                
                # Prepare messages - just system prompt and current request
                # DON'T include conversation history to avoid tool_call/tool_result mismatch
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=agent_message))
                
                # Create model with tools (including our Complete tool)
                enhanced_tools = self._create_enhanced_tools(tools)
                model_with_tools = model.bind_tools(enhanced_tools)
                
                # Get response
                response = await model_with_tools.ainvoke(messages)
                
                # Check if agent called Complete tool
                is_complete = False
                summary = ""
                
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        if tool_call.get('name') == 'Complete':
                            is_complete = True
                            break
                
                logger.info(f"Agent loop {state['loop_step'] + 1}: {'Completing' if is_complete else 'Continuing'}")
                
                return {
                    "messages": [response],  # Keep the agent response for conversation history
                    "loop_step": state["loop_step"] + 1,
                    "is_complete": is_complete,
                    "current_progress": state["current_progress"]
                }
                
            except Exception as e:
                logger.error(f"Error in agent node: {e}")
                return {
                    "messages": [AIMessage(content=f"Error: {str(e)}")],
                    "loop_step": state["loop_step"] + 1,
                    "is_complete": True  # Stop on error
                }
        
        async def tool_node(state: ReflectionState) -> Dict[str, Any]:
            """Execute tool calls"""
            last_message = state["messages"][-1]
            tool_results = []
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    if tool_call.get('name') == 'Complete':
                        # Handle completion
                        tool_results.append(
                            ToolMessage(
                                content="Task completed",
                                tool_call_id=tool_call.get('id', 'complete'),
                                name='Complete'
                            )
                        )
                        continue
                    
                    # Execute regular tools
                    try:
                        tool = next((t for t in tools if getattr(t, 'name', str(t)) == tool_call.get('name')), None)
                        if tool:
                            # Log the actual tool call arguments that are being executed
                            tool_args = tool_call.get('args', {})
                            logger.info(f"Executing tool '{tool_call.get('name')}' with args: {tool_args}")
                            
                            # Send real tool args via callback if available
                            if self.current_callback and tool_args:
                                try:
                                    result = self.current_callback({
                                        "node": "tool_args",
                                        "content": {
                                            "tool_call_id": tool_call.get('id'),
                                            "tool_name": tool_call.get('name'),
                                            "args": tool_args
                                        }
                                    })
                                    if hasattr(result, "__await__"):
                                        await result
                                except Exception as e:
                                    logger.error(f"Error sending tool args via callback: {e}")
                            
                            result = await tool.ainvoke(tool_args)
                            tool_results.append(
                                ToolMessage(
                                    content=str(result),
                                    tool_call_id=tool_call.get('id', 'unknown'),
                                    name=tool_call.get('name', 'unknown')
                                )
                            )
                    except Exception as e:
                        tool_results.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call.get('id', 'error'),
                                name=tool_call.get('name', 'error')
                            )
                        )
            
            # Update progress based on tool results
            progress_update = self._extract_progress(tool_results, state["current_progress"])
            
            return {
                "messages": tool_results,  # Keep tool results for conversation history
                "current_progress": progress_update
            }
        
        async def reflection_node(state: ReflectionState) -> Dict[str, Any]:
            """Evaluate if work is complete"""
            try:
                # Extract work completed from messages
                work_completed = self._extract_work_summary(state["messages"])
                
                reflection_message = self.reflection_prompt.format(
                    original_request=state["original_request"],
                    work_completed=work_completed
                )
                
                messages = [HumanMessage(content=reflection_message)]
                response = await model.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Parse reflection response
                is_complete = content.startswith("COMPLETE:")
                
                if is_complete:
                    summary = content.replace("COMPLETE:", "").strip()
                    return {
                        "messages": [AIMessage(content=summary)],
                        "is_complete": True,
                        "current_progress": summary
                    }
                else:
                    feedback = content.replace("CONTINUE:", "").strip()
                    return {
                        "messages": [AIMessage(content=f"Need to continue: {feedback}")],
                        "is_complete": False,
                        "current_progress": state["current_progress"] + f"\nNext: {feedback}"
                    }
                    
            except Exception as e:
                logger.error(f"Error in reflection: {e}")
                return {"is_complete": True}  # Stop on error
        
        def should_continue(state: ReflectionState) -> Literal["continue", "reflect", "tools", "end"]:
            """Route based on current state"""
            last_message = state["messages"][-1]
            
            # Check loop limits
            if state["loop_step"] >= state.get("max_loops", 10):
                return "end"
            
            # Check if we're complete
            if state.get("is_complete", False):
                return "end"
            
            # If last message has tool calls, go to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Check if it's a Complete call
                for tool_call in last_message.tool_calls:
                    if tool_call.get('name') == 'Complete':
                        return "end"
                return "tools"
            
            # If we've done several loops, reflect
            if state["loop_step"] > 0 and state["loop_step"] % 3 == 0:
                return "reflect"
            
            # Otherwise continue with agent
            return "continue"
        
        # Build the graph
        workflow = StateGraph(ReflectionState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("reflect", reflection_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "agent",
                "tools": "tools", 
                "reflect": "reflect",
                "end": END
            }
        )
        
        workflow.add_edge("tools", "agent")
        workflow.add_edge("reflect", "agent")
        
        return workflow.compile()
    
    def create_react_graph(
        self,
        model: Runnable,
        tools: Sequence[Any],
        system_prompt: str = "",
        checkpointer = None
    ) -> CompiledStateGraph:
        """Create a standard ReAct agent graph"""
        return create_react_agent(
            model,
            tools,
            checkpointer=checkpointer,
            prompt=system_prompt
        )
    
    async def astream_graph(
        self,
        graph: CompiledStateGraph,
        inputs: dict,
        config: Optional[RunnableConfig] = None,
        callback: Optional[Callable] = None,
        graph_type: str = "simple"
    ) -> Dict[str, Any]:
        """
        Unified streaming for both ReAct and Reflection graphs
        Handles agent->tools->complete pattern with optional reflect node
        """
        config = config or {}
        final_result = {}
        collected_content = []
        
        # Store callback for tool nodes to access
        self.current_callback = callback
        
        # Use messages mode for both graph types for consistency
        async for chunk_msg, metadata in graph.astream(
            inputs, config, stream_mode="messages"
        ):
            curr_node = metadata["langgraph_node"]
            
            # Handle different node types with unified pattern
            if curr_node == 'agent':
                # Extract text content from agent responses
                if hasattr(chunk_msg, 'type') and chunk_msg.type == 'AIMessageChunk':
                    content_str = str(getattr(chunk_msg, 'content', ''))
                    
                    if content_str and content_str not in ['[]', '']:
                        # Try to parse structured content
                        if content_str.startswith('[') and content_str.endswith(']'):
                            try:
                                import ast
                                parsed_content = ast.literal_eval(content_str)
                                if isinstance(parsed_content, list):
                                    for content_block in parsed_content:
                                        if isinstance(content_block, dict) and content_block.get('type') == 'text':
                                            text_content = content_block.get('text', '')
                                            if text_content:
                                                collected_content.append(text_content)
                            except Exception:
                                if content_str.strip():
                                    collected_content.append(content_str)
                        else:
                            collected_content.append(content_str)
            
            elif curr_node == 'tools':
                # Handle tool execution - standard for both graph types
                if hasattr(chunk_msg, 'content'):
                    tool_content = f"Tool executed: {getattr(chunk_msg, 'name', 'Unknown')}\nResult: {chunk_msg.content}"
                    collected_content.append(tool_content)
            
            elif curr_node == 'reflect' and graph_type == 'reflection':
                # Handle reflection node - only for reflection graphs
                if hasattr(chunk_msg, 'content'):
                    reflection_content = f"Reflection: {chunk_msg.content}"
                    collected_content.append(reflection_content)
            
            # Update final result
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
        
        # Add collected content to final result
        if collected_content:
            final_result["collected_content"] = "".join(collected_content)
        else:
            # Fallback to extracting from final message
            if hasattr(final_result.get("content"), "content"):
                final_result["collected_content"] = str(final_result["content"].content)
        
        return final_result
    
    def _create_enhanced_tools(self, tools: Sequence[Any]) -> List[Dict[str, Any]]:
        """Add Complete tool to the tool list"""
        enhanced_tools = list(tools)
        
        # Add Complete tool
        complete_tool = {
            "type": "function",
            "function": {
                "name": "Complete",
                "description": "Call this when you have finished helping the user (after providing a summary in your response)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        enhanced_tools.append(complete_tool)
        return enhanced_tools
    
    def _extract_progress(self, tool_results: List[ToolMessage], current_progress: str) -> str:
        """Extract progress from tool results - no truncation, full context"""
        if not tool_results:
            return current_progress
        
        # Keep all tool results with full content - no truncation
        results_summary = []
        for result in tool_results:
            results_summary.append(f"Used {result.name}: {result.content}")
        
        new_progress = "\n".join(results_summary)
        
        # Accumulate full progress - no truncation
        if current_progress:
            return f"{current_progress}\n{new_progress}"
        return new_progress
    
    def _extract_work_summary(self, messages: List[BaseMessage]) -> str:
        """Extract a summary of work completed from messages - full context"""
        recent_messages = messages[-20:]  # Get more messages for full context
        
        summary_parts = []
        for msg in recent_messages:
            if isinstance(msg, ToolMessage):
                # Keep full content - no truncation
                summary_parts.append(f"Tool {msg.name}: {msg.content}")
            elif isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                content = str(msg.content)
                if content:
                    summary_parts.append(f"Agent: {content}")
        
        return "\n".join(summary_parts)  # Return all parts