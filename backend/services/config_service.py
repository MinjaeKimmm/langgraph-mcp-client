import json
import os
from typing import Dict, Any
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import ToolConfig


class ConfigService:
    def __init__(self, config_file: str = "mcp_servers/config.json"):
        self.config_file = Path(config_file)
        self.default_config = {
            "get_current_time": {
                "command": "python", 
                "args": ["./mcp_servers/time_server.py"],
                "transport": "stdio"
            }
        }

    def load_config(self) -> Dict[str, Any]:
        """Load MCP configuration from JSON file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                # Create file with default config if it doesn't exist
                self.save_config(self.default_config)
                return self.default_config
        except Exception as e:
            raise RuntimeError(f"Error loading config file: {str(e)}")

    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save MCP configuration to JSON file"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            raise RuntimeError(f"Error saving config file: {str(e)}")

    def add_tool(self, tool_name: str, tool_config: ToolConfig) -> Dict[str, Any]:
        """Add a new tool to configuration"""
        config = self.load_config()
        config[tool_name] = tool_config.model_dump(exclude_none=True)
        self.save_config(config)
        return config

    def remove_tool(self, tool_name: str) -> Dict[str, Any]:
        """Remove a tool from configuration"""
        config = self.load_config()
        if tool_name in config:
            del config[tool_name]
            self.save_config(config)
        return config

    def validate_tool_config(self, tool_config: ToolConfig) -> bool:
        """Validate tool configuration"""
        # Check required fields
        if not tool_config.command and not tool_config.url:
            raise ValueError("Tool configuration requires either 'command' or 'url' field")
        
        if tool_config.command and not tool_config.args:
            raise ValueError("Tool configuration with 'command' requires 'args' field")
        
        if tool_config.command and not isinstance(tool_config.args, list):
            raise ValueError("'args' field must be a list")
        
        # Set default transport
        if tool_config.url and not tool_config.transport:
            tool_config.transport = "sse"
        elif not tool_config.transport:
            tool_config.transport = "stdio"
            
        return True

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace entire configuration"""
        # Validate all tools in new config
        for tool_name, tool_data in new_config.items():
            tool_config = ToolConfig(**tool_data)
            self.validate_tool_config(tool_config)
        
        self.save_config(new_config)
        return new_config