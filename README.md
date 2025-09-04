# LangGraph MCP Agents

A modern, modular system for running LangGraph ReAct agents with Model Context Protocol (MCP) tool integration. This project separates the original Streamlit monolith into a clean FastAPI backend and React TypeScript frontend.

## 🏗️ Architecture

```
langgraph-mcp-agents/
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main application with streaming chat API
│   ├── models.py              # Pydantic models for API
│   ├── services/              # Service layer
│   │   ├── config_service.py  # MCP configuration management
│   │   ├── mcp_service.py     # MCP client and tool management
│   │   └── agent_service.py   # LangGraph agent orchestration
│   ├── mcp_servers/           # MCP tool configurations
│   │   └── config.json        # Tool definitions and settings
│   └── requirements.txt       # Python dependencies
├── frontend/                   # React TypeScript frontend
│   ├── src/App.tsx            # Main chat interface component
│   ├── package.json           # Node.js dependencies
│   └── README.md              # Frontend-specific documentation
└── README.md                  # This file
```

## ✨ Features

### Backend (FastAPI)
- **🔄 Streaming Chat**: Real-time Server-Sent Events for live conversation
- **🛠️ MCP Integration**: Dynamic tool management via Model Context Protocol
- **🤖 Multi-Model Support**: Claude (Anthropic) and GPT (OpenAI) models
- **⚙️ Configuration API**: Add/remove tools dynamically via REST endpoints
- **🧵 Thread Management**: Persistent conversation contexts
- **📊 Health Monitoring**: Service status and diagnostics

### Frontend (React)
- **💬 Real-time Chat**: Smooth streaming text display with tool call separation
- **🛠️ Dynamic Tool Management**: Add/remove MCP tools with JSON configuration
- **⚙️ Model Selection**: Switch between Claude and GPT models seamlessly
- **🔧 Advanced Settings**: Timeout, recursion limits, and thread management
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🎨 Modern UI**: Tailwind CSS with Lucide React icons

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Environment variables for model APIs (see `.env` setup)

### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here

# Start FastAPI server
python app.py
```

Backend runs on `http://localhost:8000`

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend runs on `http://localhost:3000`

## 📋 Usage

### Basic Chat
1. Start both backend and frontend servers
2. Open `http://localhost:3000` in your browser
3. Type messages and press Enter to chat with the agent
4. Watch real-time streaming responses with tool call details

### Managing MCP Tools
1. **View Tools**: Check the sidebar for currently available tools
2. **Add Tools**: Click "+" and provide tool name and JSON configuration:
   ```json
   {
     "command": "python",
     "args": ["-m", "your_mcp_server"],
     "transport": "stdio"
   }
   ```
3. **Remove Tools**: Click trash icon next to any tool
4. **Tool Calls**: Expand tool sections in chat to see inputs/outputs

### Configuration Options
- **Model Selection**: Choose between Claude 3.5 Sonnet, Claude 3 Haiku, GPT-4, etc.
- **Timeout**: Set request timeout (default: 120 seconds)
- **Recursion Limit**: Control agent reasoning depth (default: 100)
- **Thread Management**: Reset conversation or continue existing threads

## 🔧 MCP Tool Integration

This system uses the Model Context Protocol to integrate external tools:

### Built-in Tools
- **Time/Date**: Get current time in various timezones
- **File Operations**: Read/write files (configured in `mcp_servers/config.json`)
- **Web Search**: Internet search capabilities (if configured)

### Adding Custom Tools
1. Create your MCP server following [MCP specification](https://modelcontextprotocol.io)
2. Add configuration via the frontend UI or directly edit `mcp_servers/config.json`
3. Tools are loaded dynamically without server restart

### Example Tool Configuration
```json
{
  "datetime": {
    "command": "python",
    "args": ["-m", "mcp_servers.datetime"],
    "transport": "stdio"
  },
  "filesystem": {
    "command": "python",
    "args": ["-m", "mcp_servers.filesystem", "/allowed/path"],
    "transport": "stdio"
  }
}
```

## 🔍 API Reference

### Key Endpoints
- `GET /status` - Agent and system status
- `GET /tools` - List available MCP tools
- `POST /chat` - Send message (non-streaming)
- `POST /chat/stream` - Send message with real-time streaming
- `GET /config` - Get MCP tool configuration
- `POST /config` - Update MCP configuration
- `POST /config/tool` - Add single tool
- `DELETE /config/tool/{name}` - Remove tool

### Streaming Format
Server-Sent Events with JSON payloads:
```javascript
{
  "type": "text" | "tool" | "complete" | "error",
  "content": "message content",
  "is_complete": boolean
}
```

## 🛠️ Development

### Backend Development
- Service-oriented architecture with clear separation of concerns
- Async/await patterns throughout for performance
- Pydantic models for type safety and validation
- Error handling and logging for debugging

### Frontend Development
- TypeScript for type safety
- React 19 with modern hooks and patterns
- Tailwind CSS for consistent styling
- Real-time updates with flushSync for smooth UX

### Testing
```bash
# Backend
cd backend
python -m pytest

# Frontend  
cd frontend
npm test
```