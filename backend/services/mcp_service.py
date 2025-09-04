import asyncio
from typing import Dict, List, Any, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from services.config_service import ConfigService
from models import ToolInfo


class MCPService:
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        self.client: Optional[MultiServerMCPClient] = None
        self._tools: List[Any] = []
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize MCP client with current configuration"""
        try:
            await self.cleanup()
            
            config = self.config_service.load_config()
            if not config:
                return False
            
            self.client = MultiServerMCPClient(config)
            self._tools = await self.client.get_tools()
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing MCP service: {str(e)}")
            return False

    async def cleanup(self):
        """Clean up existing client connection"""
        if self.client is not None:
            self.client = None
        self._tools = []
        self._initialized = False

    def get_tools(self) -> List[Any]:
        """Get list of available tools from all MCP servers"""
        return self._tools

    def get_tool_info(self) -> List[ToolInfo]:
        """Get detailed information about available tools"""
        tool_info = []
        for tool in self._tools:
            info = ToolInfo(
                name=tool.name,
                description=getattr(tool, 'description', None),
                parameters=getattr(tool, 'args_schema', None),
                server_name="unknown"  # Could be enhanced to track server names
            )
            tool_info.append(info)
        return tool_info

    def get_tool_count(self) -> int:
        """Get number of available tools"""
        return len(self._tools)

    def is_initialized(self) -> bool:
        """Check if MCP service is properly initialized"""
        return self._initialized and self.client is not None

    async def add_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> bool:
        """Add a new tool and reinitialize client"""
        try:
            from ..models import ToolConfig
            
            # Validate and add tool to config
            validated_config = ToolConfig(**tool_config)
            self.config_service.validate_tool_config(validated_config)
            self.config_service.add_tool(tool_name, validated_config)
            
            # Reinitialize client with new config
            return await self.initialize()
            
        except Exception as e:
            print(f"Error adding tool {tool_name}: {str(e)}")
            return False

    async def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool and reinitialize client"""
        try:
            self.config_service.remove_tool(tool_name)
            return await self.initialize()
        except Exception as e:
            print(f"Error removing tool {tool_name}: {str(e)}")
            return False

    async def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update entire configuration and reinitialize client"""
        try:
            self.config_service.update_config(new_config)
            return await self.initialize()
        except Exception as e:
            print(f"Error updating config: {str(e)}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """Get current MCP configuration"""
        return self.config_service.load_config()