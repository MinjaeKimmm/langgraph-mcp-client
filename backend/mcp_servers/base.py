import os
import logging
from abc import ABC, abstractmethod
from fastmcp import FastMCP

class BaseMCPServer(ABC):
    """Base class for MCP servers using FastMCP"""
    
    def __init__(self, name: str, version: str = "0.1.0", instructions: str = ""):
        """Initialize the MCP server"""
        self.name = name
        self.version = version
        self.mcp = FastMCP(name=name, instructions=instructions)
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        self.setup_handlers()
    
    def tool(self, **kwargs):
        """Decorator for tools that automatically adds server name to metadata"""
        # Add server name to tool metadata
        if 'tags' not in kwargs:
            kwargs['tags'] = set()
        if isinstance(kwargs['tags'], list):
            kwargs['tags'] = set(kwargs['tags'])
        kwargs['tags'].add(f"server:{self.name}")
        
        return self.mcp.tool(**kwargs)
    
    @abstractmethod
    def setup_handlers(self):
        """Setup all MCP handlers - must be implemented by subclasses"""
        pass
    
    def run(self, transport: str = "stdio", host: str = None, port: int = None):
        """Run the MCP server"""
        # Suppress FastMCP banner by setting environment variable
        os.environ['FASTMCP_QUIET'] = '1'
        if transport == "http":
            if not host or not port:
                raise ValueError("Host and port must be specified for HTTP transport")
            self.logger.info(f"Starting {self.name} HTTP server on {host}:{port}")
            self.mcp.run(transport="http", show_banner=False)
        else:
            self.logger.info(f"Starting {self.name} stdio server")
            self.mcp.run(transport="stdio", show_banner=False)