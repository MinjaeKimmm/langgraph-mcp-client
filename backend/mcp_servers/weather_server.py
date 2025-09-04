import os
from typing import Optional, Dict, Any
from mcp_servers.base import BaseMCPServer

# Optional: Import weather API libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("requests not installed. Using mock weather data.")


class WeatherMCPServer(BaseMCPServer):
    """Weather MCP Server implementation for weather information"""
    
    def __init__(self):
        super().__init__(
            name="Weather",
            version="0.1.0",
            instructions="You are a weather assistant that can answer questions about the weather in a given location."
        )
        self.api_key = os.environ.get('OPENWEATHER_API_KEY')
        self.use_real_api = REQUESTS_AVAILABLE and self.api_key
        
        if not self.use_real_api:
            self.logger.info("Using mock weather data. Set OPENWEATHER_API_KEY for real weather.")
    
    def setup_handlers(self):
        """Setup all MCP handlers for weather operations"""
        
        @self.tool()
        async def get_weather(location: str) -> str:
            """
            Get current weather information for the specified location.

            This function can either use a real weather API (OpenWeatherMap) if configured,
            or return mock data for demonstration purposes.

            Args:
                location (str): The name of the location (city, region, etc.) to get weather for

            Returns:
                str: A string containing the weather information for the specified location
            """
            return await self._get_weather(location)
        
        @self.tool()
        async def get_forecast(location: str, days: Optional[int] = 3) -> str:
            """
            Get weather forecast for the specified location.
            
            Args:
                location (str): The location to get forecast for
                days (int, optional): Number of days to forecast (1-5). Defaults to 3.
                
            Returns:
                str: Weather forecast information
            """
            return await self._get_forecast(location, days)
    
    async def _get_weather(self, location: str) -> str:
        """Internal method to get current weather"""
        if self.use_real_api:
            return await self._get_real_weather(location)
        else:
            return await self._get_mock_weather(location)
    
    async def _get_real_weather(self, location: str) -> str:
        """Get real weather data from OpenWeatherMap API"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']
            description = data['weather'][0]['description'].title()
            
            return f"Weather in {location}:\n" \
                   f"Temperature: {temp}°C (feels like {feels_like}°C)\n" \
                   f"Conditions: {description}\n" \
                   f"Humidity: {humidity}%"
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {str(e)}"
        except KeyError as e:
            return f"Error parsing weather data: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    async def _get_mock_weather(self, location: str) -> str:
        """Return mock weather data"""
        # Mock weather responses with some variety
        import random
        
        conditions = [
            ("Sunny", "25°C", "Low humidity, perfect day!"),
            ("Partly Cloudy", "22°C", "Comfortable with some clouds"),
            ("Rainy", "18°C", "Light rain expected"),
            ("Overcast", "20°C", "Cloudy but pleasant"),
            ("Sunny", "28°C", "Warm and bright")
        ]
        
        condition, temp, description = random.choice(conditions)
        
        return f"Mock Weather in {location}:\n" \
               f"Temperature: {temp}\n" \
               f"Conditions: {condition}\n" \
               f"Note: {description}\n" \
               f"(This is mock data - set OPENWEATHER_API_KEY for real weather)"
    
    async def _get_forecast(self, location: str, days: Optional[int]) -> str:
        """Internal method to get weather forecast"""
        if days is None:
            days = 3
        days = max(1, min(days, 5))  # Limit to 1-5 days
        
        if self.use_real_api:
            return await self._get_real_forecast(location, days)
        else:
            return await self._get_mock_forecast(location, days)
    
    async def _get_real_forecast(self, location: str, days: int) -> str:
        """Get real forecast data from OpenWeatherMap API"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # API returns 3-hour intervals, 8 per day
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Group by day and get daily summary
            daily_forecasts = []
            current_date = None
            day_data = []
            
            for item in data['list']:
                date = item['dt_txt'].split(' ')[0]
                if current_date != date:
                    if day_data:
                        # Process previous day
                        avg_temp = sum(d['main']['temp'] for d in day_data) / len(day_data)
                        conditions = day_data[len(day_data)//2]['weather'][0]['description'].title()
                        daily_forecasts.append(f"{current_date}: {avg_temp:.1f}°C, {conditions}")
                    
                    current_date = date
                    day_data = []
                
                day_data.append(item)
            
            # Process last day
            if day_data:
                avg_temp = sum(d['main']['temp'] for d in day_data) / len(day_data)
                conditions = day_data[len(day_data)//2]['weather'][0]['description'].title()
                daily_forecasts.append(f"{current_date}: {avg_temp:.1f}°C, {conditions}")
            
            return f"{days}-day forecast for {location}:\n" + "\n".join(daily_forecasts[:days])
            
        except Exception as e:
            return f"Error fetching forecast: {str(e)}"
    
    async def _get_mock_forecast(self, location: str, days: int) -> str:
        """Return mock forecast data"""
        import random
        from datetime import datetime, timedelta
        
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]
        
        forecasts = []
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            temp = random.randint(15, 30)
            condition = random.choice(conditions)
            forecasts.append(f"{date}: {temp}°C, {condition}")
        
        return f"Mock {days}-day forecast for {location}:\n" + "\n".join(forecasts) + \
               "\n(This is mock data - set OPENWEATHER_API_KEY for real weather)"
    
    def start(self, transport: str = "stdio", host: str = None, port: int = None):
        """Start the server"""
        self.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    server = WeatherMCPServer()
    server.start(transport="stdio")