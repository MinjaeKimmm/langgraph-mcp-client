from datetime import datetime
import pytz
from typing import Optional
from mcp_servers.base import BaseMCPServer


class TimeMCPServer(BaseMCPServer):
    """Time MCP Server implementation for timezone and datetime operations"""
    
    def __init__(self):
        super().__init__(
            name="TimeService",
            version="0.1.0",
            instructions="You are a time assistant that can provide the current time for different timezones."
        )
    
    def setup_handlers(self):
        """Setup all MCP handlers for time operations"""
        
        @self.tool()
        async def get_current_time(timezone: Optional[str] = "Asia/Seoul") -> str:
            """
            Get current time information for the specified timezone.

            This function returns the current system time for the requested timezone.

            Args:
                timezone (str, optional): The timezone to get current time for. Defaults to "Asia/Seoul".

            Returns:
                str: A string containing the current time information for the specified timezone
            """
            return await self._get_current_time(timezone)
        
        @self.tool()
        async def list_timezones(region: Optional[str] = None) -> str:
            """
            List available timezones, optionally filtered by region.
            
            Args:
                region (str, optional): Filter timezones by region (e.g., 'Asia', 'America', 'Europe')
                
            Returns:
                str: A formatted list of available timezones
            """
            return await self._list_timezones(region)
    
    async def _get_current_time(self, timezone: str) -> str:
        """Internal method to get current time"""
        try:
            # Get the timezone object
            tz = pytz.timezone(timezone)

            # Get current time in the specified timezone
            current_time = datetime.now(tz)

            # Format the time as a string
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

            return f"Current time in {timezone} is: {formatted_time}"
        except pytz.exceptions.UnknownTimeZoneError:
            return f"Error: Unknown timezone '{timezone}'. Please provide a valid timezone."
        except Exception as e:
            return f"Error getting time: {str(e)}"
    
    async def _list_timezones(self, region: Optional[str] = None) -> str:
        """Internal method to list timezones"""
        try:
            all_timezones = list(pytz.all_timezones)
            
            if region:
                # Filter by region prefix
                filtered_timezones = [tz for tz in all_timezones if tz.startswith(region + '/')]
                if not filtered_timezones:
                    return f"No timezones found for region '{region}'. Try: Asia, America, Europe, Africa, etc."
                timezones = sorted(filtered_timezones)[:20]  # Limit to 20 for readability
                return f"Timezones in {region}:\n" + "\n".join(f"- {tz}" for tz in timezones)
            else:
                # Show common timezones
                common_timezones = [
                    'UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific',
                    'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Europe/Rome',
                    'Asia/Tokyo', 'Asia/Seoul', 'Asia/Shanghai', 'Asia/Dubai',
                    'Australia/Sydney', 'Australia/Melbourne'
                ]
                return "Common timezones:\n" + "\n".join(f"- {tz}" for tz in common_timezones)
        except Exception as e:
            return f"Error listing timezones: {str(e)}"
    
    def start(self, transport: str = "stdio", host: str = None, port: int = None):
        """Start the server"""
        self.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    server = TimeMCPServer()
    server.start(transport="stdio")