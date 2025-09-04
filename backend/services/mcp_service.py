import logging
from typing import Any
from models import ToolInfo, ServerToolInfo, GroupedToolsResponse, ToolConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from .config_service import ConfigService

logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self, config_service: ConfigService):
        self.config_service: ConfigService = config_service
        self.client: MultiServerMCPClient | None = None
        self._initialized: bool = False

    async def initialize(self) -> bool:
        """Initialize MCP client with current configuration"""
        try:
            await self.cleanup()
            
            config = self.config_service.load_config()
            if not config:
                return False
            
            # Create single MultiServerMCP client - it handles all servers efficiently
            self.client = MultiServerMCPClient(config)
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MCP service: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    async def cleanup(self):
        """Clean up existing client connection"""
        if self.client is not None:
            self.client = None
        self._initialized = False

    async def get_tools(self) -> list[Any]:
        """Get all available tools from all servers"""
        if not self.client:
            return []
        return await self.client.get_tools()

    async def get_grouped_tools(self) -> GroupedToolsResponse:
        """Get tools grouped by their server"""
        if not self.client:
            return GroupedToolsResponse(servers={})
            
        config = self.config_service.load_config()
        servers = {}
        
        for server_name in config.keys():
            try:
                # Get tools from this specific server using efficient API
                server_tools = await self.client.get_tools(server_name=server_name)
                
                tool_infos = []
                for tool in server_tools:
                    tool_info = ToolInfo(
                        name=tool.name,
                        description=getattr(tool, 'description', None),
                        parameters=getattr(tool, 'args_schema', None),
                        server_name=server_name
                    )
                    tool_infos.append(tool_info)
                
                servers[server_name] = ServerToolInfo(
                    name=self._get_server_display_name(server_name),
                    description=None,
                    tools=tool_infos
                )
                
            except Exception as e:
                logger.warning(f"Failed to get tools from server '{server_name}': {e}")
        
        return GroupedToolsResponse(servers=servers)
    
    def _get_server_display_name(self, server_name: str) -> str:
        """Convert server name to display name"""
        name_mapping = {
            'time': 'Time Service',
            'weather': 'Weather Service', 
            'google_drive': 'Google Drive',
            'googledrive': 'Google Drive'
        }
        return name_mapping.get(server_name.lower(), server_name.replace('_', ' ').title())

    async def get_tool_count(self) -> int:
        """Get number of available tools"""
        if not self.client:
            return 0
        tools = await self.client.get_tools()
        return len(tools)

    def is_initialized(self) -> bool:
        """Check if MCP service is properly initialized"""
        return self._initialized and self.client is not None

    async def add_tool(self, tool_name: str, tool_config: dict[str, Any]) -> bool:
        """Add a new tool and reinitialize client"""
        try:            
            # Validate and add tool to config
            validated_config = ToolConfig(**tool_config)
            self.config_service.validate_tool_config(validated_config)
            _ = self.config_service.add_tool(tool_name, validated_config)
            
            # Reinitialize client with new config
            return await self.initialize()
            
        except Exception as e:
            logger.error(f"Error adding tool {tool_name}: {str(e)}")
            return False

    async def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool and reinitialize client"""
        try:
            _ = self.config_service.remove_tool(tool_name)
            return await self.initialize()
        except Exception as e:
            logger.error(f"Error removing tool {tool_name}: {str(e)}")
            return False

    async def update_config(self, new_config: dict[str, Any]) -> bool:
        """Update entire configuration and reinitialize client"""
        try:
            _ = self.config_service.update_config(new_config)
            return await self.initialize()
        except Exception as e:
            logger.error(f"Error updating config: {str(e)}")
            return False

    def get_config(self) -> dict[str, Any]:
        """Get current MCP configuration"""
        return self.config_service.load_config()
    
    async def get_filtered_tools(self, enabled_tools: list[str] | None = None) -> list[Any]:
        """Get tools filtered by enabled server names"""
        if not self.client or not enabled_tools:
            return await self.get_tools()
        
        filtered_tools = []
        for server_name in enabled_tools:
            try:
                server_tools = await self.client.get_tools(server_name=server_name)
                filtered_tools.extend(server_tools)
            except Exception as e:
                logger.warning(f"Failed to get tools from server '{server_name}': {e}")
        
        return filtered_tools
    
    async def test_and_add_tool(self, tool_name: str, tool_config: dict[str, Any]) -> bool:
        """Test MCP connection and only save config if successful"""
        try:
            # Create temporary config with just this tool for testing
            test_config = {tool_name: tool_config}
            
            # Create temporary client to test connection
            from langchain_mcp_adapters.client import MultiServerMCPClient
            test_client = MultiServerMCPClient(test_config)
            
            # Try to get tools from this server to test connection
            server_tools = await test_client.get_tools(server_name=tool_name)
            
            # Connection successful - now add to actual config
            validated_config = ToolConfig(**tool_config)
            self.config_service.validate_tool_config(validated_config)
            _ = self.config_service.add_tool(tool_name, validated_config)
            
            # Reinitialize our main client with the new config
            return await self.initialize()
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{tool_name}': {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
