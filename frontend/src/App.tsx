import React, { useState, useEffect, useRef } from 'react';
import { flushSync } from 'react-dom';
import axios from 'axios';
import { Send, Settings, RefreshCw, Plus, Bot, User, Loader2, ChevronDown, ChevronRight, Wrench } from 'lucide-react';
import './App.css';

// Types matching backend models
interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
  toolCalls?: string; // For storing tool call information
}

interface ChatRequest {
  message: string;
  model?: string;
  timeout_seconds?: number;
  recursion_limit?: number;
  thread_id?: string;
  enabled_tools?: string[];
}

interface ChatResponse {
  response: string;
  model_used: string;
  execution_time: number;
  thread_id: string;
}

interface StreamingChatResponse {
  type: 'text' | 'tool' | 'complete' | 'error';
  content: string;
  is_complete: boolean;
}

interface ToolConfig {
  command?: string;
  args?: string[];
  url?: string;
  transport: 'stdio' | 'sse';
}

interface AgentStatus {
  initialized: boolean;
  tool_count: number;
  model: string;
  available_models: string[];
}

interface ToolInfo {
  name: string;
  description?: string;
  parameters?: any;
}

interface ServerInfo {
  name: string;
  description?: string;
  tools: ToolInfo[];
}

interface GroupedToolsResponse {
  servers: Record<string, ServerInfo>;
}

const API_BASE_URL = 'http://localhost:8000';

function App() {
  // State management
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [groupedTools, setGroupedTools] = useState<GroupedToolsResponse>({ servers: {} });
  // const [config, setConfig] = useState<Record<string, ToolConfig>>({});
  const [selectedModel, setSelectedModel] = useState('claude-3-7-sonnet-latest');
  const [timeoutSeconds, setTimeoutSeconds] = useState(120);
  const [recursionLimit, setRecursionLimit] = useState(100);
  const [threadId, setThreadId] = useState<string>('');
  const [enabledTools, setEnabledTools] = useState<Set<string>>(new Set());
  
  // UI state
  const [showSettings, setShowSettings] = useState(false);
  const [newToolName, setNewToolName] = useState('');
  const [newToolConfig, setNewToolConfig] = useState('{}');
  const [expandedTools, setExpandedTools] = useState<{[key: string]: boolean}>({});
  const [showToolPanel, setShowToolPanel] = useState(true); // Open by default
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initial setup
  useEffect(() => {
    fetchStatus();
    fetchTools();
    // Generate initial thread ID
    setThreadId(Math.random().toString(36).substring(2, 15));
  }, []);
  
  // Enable all tools by default when tools are loaded
  useEffect(() => {
    const serverNames = Object.keys(groupedTools.servers);
    if (serverNames.length > 0 && enabledTools.size === 0) {
      setEnabledTools(new Set(serverNames));
    }
  }, [groupedTools, enabledTools.size]);

  // API calls
  const fetchStatus = async () => {
    try {
      const response = await axios.get<AgentStatus>(`${API_BASE_URL}/status`);
      setStatus(response.data);
      setSelectedModel(response.data.model);
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  };

  const fetchTools = async () => {
    try {
      const response = await axios.get<GroupedToolsResponse>(`${API_BASE_URL}/tools`);
      setGroupedTools(response.data);
    } catch (error) {
      console.error('Error fetching tools:', error);
    }
  };


  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    const originalMessage = inputMessage;
    setInputMessage('');
    setIsLoading(true);

    // Create a unique timestamp identifier for this assistant message
    const assistantTimestamp = new Date().toISOString();
    
    // Create a placeholder assistant message immediately for the "thinking" state
    const thinkingMessage: ChatMessage = {
      role: 'assistant',
      content: '', // Empty content initially
      timestamp: assistantTimestamp,
      toolCalls: ''
    };
    
    setMessages(prev => [...prev, thinkingMessage]);

    try {
      const request: ChatRequest = {
        message: originalMessage,
        model: selectedModel,
        timeout_seconds: timeoutSeconds,
        recursion_limit: recursionLimit,
        thread_id: threadId,
        enabled_tools: Array.from(enabledTools)
      };

      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Failed to get response reader');
      }

      const decoder = new TextDecoder();
      let accumulatedContent = '';
      let accumulatedToolCalls = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data.trim()) {
              try {
                const parsed: StreamingChatResponse = JSON.parse(data);
                
                if (parsed.type === 'text' && parsed.content) {
                  // Update existing assistant message with text content
                  accumulatedContent += parsed.content;
                  
                  // Force immediate UI update using flushSync
                  flushSync(() => {
                    setMessages(prev => 
                      prev.map(msg => 
                        msg.timestamp === assistantTimestamp && msg.role === 'assistant'
                          ? {
                              ...msg,
                              content: accumulatedContent,
                              toolCalls: accumulatedToolCalls
                            }
                          : msg
                      )
                    );
                  });
                  
                  // Small delay to ensure UI renders before processing next chunk
                  await new Promise(resolve => setTimeout(resolve, 10));
                } else if (parsed.type === 'tool' && parsed.content) {
                  // Update existing assistant message with tool calls
                  accumulatedToolCalls += (accumulatedToolCalls ? '\n\n' : '') + parsed.content;
                  
                  // Update tool calls with immediate UI update
                  flushSync(() => {
                    setMessages(prev => 
                      prev.map(msg => 
                        msg.timestamp === assistantTimestamp && msg.role === 'assistant'
                          ? {
                              ...msg,
                              content: accumulatedContent,
                              toolCalls: accumulatedToolCalls
                            }
                          : msg
                      )
                    );
                  });
                } else if (parsed.type === 'complete') {
                  // Final completion signal
                  break;
                } else if (parsed.type === 'error') {
                  throw new Error(parsed.content);
                }

                if (parsed.is_complete) {
                  break;
                }
              } catch (parseError) {
                console.error('Error parsing streaming data:', parseError);
              }
            }
          }
        }
      }
    } catch (error: any) {
      // Update existing assistant message with error
      setMessages(prev => 
        prev.map(msg => 
          msg.timestamp === assistantTimestamp && msg.role === 'assistant'
            ? { 
                ...msg, 
                content: `Error: ${error.message || 'An error occurred'}` 
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const addTool = async () => {
    if (!newToolName.trim()) return;

    try {
      const toolConfig: ToolConfig = JSON.parse(newToolConfig);
      await axios.post(`${API_BASE_URL}/config/tool?tool_name=${newToolName}`, toolConfig);
      
      // Refresh data
      await Promise.all([fetchStatus(), fetchTools()]);
      
      setNewToolName('');
      setNewToolConfig('{}');
    } catch (error: any) {
      alert(`Error adding tool: ${error.response?.data?.detail || error.message}`);
    }
  };

  const removeTool = async (toolName: string) => {
    try {
      await axios.delete(`${API_BASE_URL}/config/tool/${toolName}`);
      
      // Refresh data
      await Promise.all([fetchStatus(), fetchTools()]);
    } catch (error: any) {
      alert(`Error removing tool: ${error.response?.data?.detail || error.message}`);
    }
  };

  const resetConversation = () => {
    setMessages([]);
    setThreadId(Math.random().toString(36).substring(2, 15));
  };

  // Tool selection functions
  const toggleTool = (serverName: string) => {
    setEnabledTools(prev => {
      const newSet = new Set(prev);
      if (newSet.has(serverName)) {
        newSet.delete(serverName);
      } else {
        newSet.add(serverName);
      }
      return newSet;
    });
  };

  const enableAllTools = () => {
    const serverNames = Object.keys(groupedTools.servers);
    setEnabledTools(new Set(serverNames));
  };

  const disableAllTools = () => {
    setEnabledTools(new Set());
  };

  // Get unique servers (tool groups)
  const getUniqueServers = () => {
    return Object.entries(groupedTools.servers).map(([serverId, serverInfo]) => ({
      id: serverId,
      name: serverInfo.name,
      tools: serverInfo.tools
    }));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-80 bg-white shadow-lg flex flex-col">
        {/* Header */}
        <div className="p-4 border-b">
          <h1 className="text-xl font-bold text-gray-800">MCP Agent</h1>
          <p className="text-sm text-gray-600">LangGraph + MCP Tools</p>
        </div>

        {/* Tool Selection Section */}
        <div className="p-4 border-b">
          <button
            onClick={() => setShowToolPanel(!showToolPanel)}
            className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-3"
          >
            {showToolPanel ? (
              <ChevronDown size={16} className="text-gray-400" />
            ) : (
              <ChevronRight size={16} className="text-gray-400" />
            )}
            <Wrench size={16} />
            Available Connectors ({Array.from(enabledTools).length}/{getUniqueServers().length})
          </button>
          
          {showToolPanel && (
            <div className="space-y-3">
              <div className="flex gap-2">
                <button
                  onClick={enableAllTools}
                  className="flex-1 px-2 py-1 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200"
                >
                  Enable All
                </button>
                <button
                  onClick={disableAllTools}
                  className="flex-1 px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200"
                >
                  Disable All
                </button>
              </div>
              
              {getUniqueServers().map(server => (
                <div key={server.id} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div className="flex-1">
                    <div className="font-medium text-sm">{server.name}</div>
                    <div className="text-xs text-gray-600">
                      {server.tools.length} tool{server.tools.length !== 1 ? 's' : ''}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {server.tools.map((tool: ToolInfo) => tool.name).join(', ')}
                    </div>
                  </div>
                  <button
                    onClick={() => toggleTool(server.id)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      enabledTools.has(server.id) 
                        ? 'bg-blue-600' 
                        : 'bg-gray-300'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        enabledTools.has(server.id) 
                          ? 'translate-x-6' 
                          : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              ))}
              
              {/* Add Connector Form */}
              <div className="space-y-2 mt-4 pt-4 border-t border-gray-200">
                <div className="text-xs font-medium text-gray-700 mb-2">Add New Connector</div>
                <input
                  placeholder="Connector name (e.g., 'my_server')"
                  value={newToolName}
                  onChange={(e) => setNewToolName(e.target.value)}
                  className="w-full p-2 border rounded text-sm"
                />
                <textarea
                  placeholder='{"command": "python", "args": ["./my_server.py"], "transport": "stdio"}'
                  value={newToolConfig}
                  onChange={(e) => setNewToolConfig(e.target.value)}
                  className="w-full p-2 border rounded text-sm h-20 resize-none"
                />
                <button
                  onClick={addTool}
                  className="w-full flex items-center justify-center gap-2 bg-blue-500 text-white p-2 rounded text-sm hover:bg-blue-600"
                >
                  <Plus size={14} />
                  Add Connector
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Settings Section */}
        <div className="p-4 border-b">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-3"
          >
            <Settings size={16} />
            System Settings
          </button>
          
          {showSettings && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Model</label>
                <select 
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-2 border rounded text-sm"
                >
                  {status?.available_models.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Timeout (seconds)</label>
                <input
                  type="number"
                  value={timeoutSeconds}
                  onChange={(e) => setTimeoutSeconds(Number(e.target.value))}
                  className="w-full p-2 border rounded text-sm"
                  min="60"
                  max="300"
                />
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Recursion Limit</label>
                <input
                  type="number"
                  value={recursionLimit}
                  onChange={(e) => setRecursionLimit(Number(e.target.value))}
                  className="w-full p-2 border rounded text-sm"
                  min="10"
                  max="200"
                />
              </div>
            </div>
          )}
        </div>


        {/* Status & Actions */}
        <div className="p-4 border-t">
          <div className="text-xs text-gray-600 mb-3">
            Status: {status?.initialized ? '✅ Ready' : '❌ Not Ready'}
            <br />
            Model: {status?.model}
            <br />
            Tools: {status?.tool_count}
          </div>
          
          <button
            onClick={resetConversation}
            className="w-full flex items-center justify-center gap-2 bg-gray-500 text-white p-2 rounded text-sm hover:bg-gray-600"
          >
            <RefreshCw size={14} />
            Reset Conversation
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white shadow-sm p-4 border-b">
          <h2 className="text-lg font-semibold text-gray-800">💬 Chat with MCP Agent</h2>
          <p className="text-sm text-gray-600">Ask questions and the agent will use available tools to help you</p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex gap-3 max-w-3xl ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  message.role === 'user' ? 'bg-blue-500' : 'bg-green-500'
                } text-white`}>
                  {message.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div className={`${
                  message.role === 'user' 
                    ? 'bg-blue-500 text-white p-3 rounded-lg' 
                    : 'space-y-2'
                }`}>
                  {message.role === 'assistant' ? (
                    <>
                      <div className="bg-white border shadow-sm p-3 rounded-lg">
                        {message.content ? (
                          <div className="whitespace-pre-wrap">{message.content}</div>
                        ) : isLoading && index === messages.length - 1 ? (
                          <div className="flex items-center gap-2 text-gray-500">
                            <Loader2 size={16} className="animate-spin" />
                            Agent is thinking...
                          </div>
                        ) : (
                          <div className="whitespace-pre-wrap">{message.content}</div>
                        )}
                        {message.timestamp && (
                          <div className="text-xs mt-2 text-gray-500">
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </div>
                        )}
                      </div>
                      {message.toolCalls && message.toolCalls.trim() && (
                        <div className="bg-gray-50 border rounded-lg overflow-hidden">
                          <button
                            onClick={() => setExpandedTools(prev => ({
                              ...prev,
                              [index]: !prev[index]
                            }))}
                            className="w-full px-3 py-2 text-left flex items-center gap-2 hover:bg-gray-100 transition-colors"
                          >
                            <Wrench size={16} className="text-orange-500" />
                            <span className="text-sm font-medium text-gray-700">Tool Call Information</span>
                            {expandedTools[index] ? (
                              <ChevronDown size={16} className="text-gray-400 ml-auto" />
                            ) : (
                              <ChevronRight size={16} className="text-gray-400 ml-auto" />
                            )}
                          </button>
                          {expandedTools[index] && (
                            <div className="px-3 pb-3 border-t bg-gray-50">
                              <pre className="text-xs text-gray-600 whitespace-pre-wrap bg-gray-100 p-2 rounded mt-2 overflow-x-auto">
                                {message.toolCalls}
                              </pre>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  ) : (
                    <>
                      <div className="whitespace-pre-wrap">{message.content}</div>
                      {message.timestamp && (
                        <div className="text-xs mt-2 text-blue-100">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
          
{/* The loading state is now handled within the assistant message itself */}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t p-4">
          <div className="flex gap-2">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Type your message here... (Enter to send, Shift+Enter for new line)"
              className="flex-1 p-3 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={2}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !inputMessage.trim()}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;