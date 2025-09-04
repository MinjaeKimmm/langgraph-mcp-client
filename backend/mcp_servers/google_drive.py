import os
import io
from typing import Optional, Dict, Any
from mcp_servers.base import BaseMCPServer

# Google API dependencies
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("google-api-python-client not installed. Google Drive server disabled.")

# Optional: Import PDF processing libraries
try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    import logging
    logging.getLogger(__name__).warning("pypdf not installed. PDF text extraction disabled.")

try:
    from markdownify import markdownify
    MARKDOWN_SUPPORT = True
except ImportError:
    MARKDOWN_SUPPORT = False


class GoogleDriveMCPServer(BaseMCPServer):
    """Google Drive MCP Server implementation"""
    
    # Google Workspace export formats
    EXPORT_FORMATS = {
        'application/vnd.google-apps.document': 'text/markdown',
        'application/vnd.google-apps.spreadsheet': 'text/csv',
        'application/vnd.google-apps.presentation': 'text/plain',
        'application/vnd.google-apps.drawing': 'image/png',
    }
    
    def __init__(self):
        super().__init__(
            name="GoogleDrive",
            version="0.1.0",
            instructions="Google Drive MCP server providing read-only access to files"
        )
        self.service = None
    
    def setup_handlers(self):
        """Setup all MCP handlers for Google Drive"""
        
        if not GOOGLE_AVAILABLE:
            # If Google APIs aren't available, provide a dummy tool
            @self.tool()
            async def google_drive_unavailable() -> str:
                """
                Google Drive functionality is not available.
                
                Returns:
                    str: Error message about missing dependencies
                """
                return "Google Drive server is not available. Please install google-api-python-client to enable this functionality."
            return
        
        @self.tool()
        async def search_files(query: str) -> str:
            """
            Search for files in Google Drive.
            
            Args:
                query (str): Search query for files
                
            Returns:
                str: List of found files
            """
            return await self._search_files(query)
        
        @self.tool()
        async def list_files(max_results: Optional[int] = 10) -> str:
            """
            List recent files from Google Drive.
            
            Args:
                max_results (int, optional): Maximum number of files to return. Defaults to 10.
                
            Returns:
                str: List of recent files
            """
            return await self._list_files(max_results)
        
        @self.tool()
        async def read_file(file_id: str) -> str:
            """
            Read content from a Google Drive file.
            
            Args:
                file_id (str): Google Drive file ID
                
            Returns:
                str: File content or error message
            """
            return await self._read_file(file_id)
    
    def get_access_token(self) -> str:
        """Get access token from environment"""
        token = os.environ.get('GDRIVE_ACCESS_TOKEN')
        if not token:
            raise RuntimeError(
                "No access token available. Set GDRIVE_ACCESS_TOKEN environment variable."
            )
        return token
    
    def initialize_service(self):
        """Initialize Google Drive service with access token"""
        if self.service:
            return  # Already initialized
        
        try:
            access_token = self.get_access_token()
            
            # Create credentials from access token
            creds = Credentials(token=access_token)
            
            # Build service
            self.service = build('drive', 'v3', credentials=creds)
            self.logger.info("Drive service initialized.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Drive service: {e}")
            raise RuntimeError(f"Failed to initialize Drive service: {e}")
    
    def _ensure_service(self):
        """Ensure Drive service is initialized"""
        if not self.service:
            self.initialize_service()
        if not self.service:
            raise RuntimeError("Drive service not initialized")
    
    async def _list_files(self, max_results: Optional[int] = None) -> str:
        """Internal method to list files"""
        if not GOOGLE_AVAILABLE:
            return "Google Drive functionality not available - missing dependencies"
        
        try:
            self._ensure_service()
            
            if max_results is None:
                max_results = 10
                
            params = {
                'pageSize': min(max_results, 50),  # Limit to 50 max
                'fields': 'files(id, name, mimeType, modifiedTime, size)'
            }
            
            results = self.service.files().list(**params).execute()
            files = results.get('files', [])
            
            if not files:
                return "No files found in Google Drive."
            
            file_list = []
            for file in files:
                name = file.get('name', 'Unnamed')
                file_id = file.get('id', 'Unknown')
                mime_type = file.get('mimeType', 'unknown')
                modified = file.get('modifiedTime', 'Unknown')
                
                file_list.append(f"• {name} ({mime_type})")
                file_list.append(f"  ID: {file_id}")
                file_list.append(f"  Modified: {modified}")
                file_list.append("")
            
            return f"Recent files from Google Drive:\n\n" + "\n".join(file_list)
            
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    async def _search_files(self, query: str) -> str:
        """Search for files in Google Drive"""
        if not GOOGLE_AVAILABLE:
            return "Google Drive functionality not available - missing dependencies"
        
        try:
            self._ensure_service()
            
            # Escape special characters
            escaped_query = query.replace('\\', '\\\\').replace("'", "\\'")
            formatted_query = f"fullText contains '{escaped_query}'"
            
            results = self.service.files().list(
                q=formatted_query,
                pageSize=10,
                fields='files(id, name, mimeType, modifiedTime, size)'
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                return f"No files found matching '{query}'"
            
            file_list = []
            for file in files:
                name = file.get('name', 'Unnamed')
                file_id = file.get('id', 'Unknown')
                mime_type = file.get('mimeType', 'unknown')
                
                file_list.append(f"• {name} ({mime_type})")
                file_list.append(f"  ID: {file_id}")
                file_list.append("")
            
            return f"Found {len(files)} files matching '{query}':\n\n" + "\n".join(file_list)
            
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    async def _read_file(self, file_id: str) -> str:
        """Internal method to read a file"""
        if not GOOGLE_AVAILABLE:
            return "Google Drive functionality not available - missing dependencies"
        
        try:
            self._ensure_service()
            
            # Get file metadata
            file = self.service.files().get(
                fileId=file_id,
                fields='mimeType, name'
            ).execute()
            
            mime_type = file.get('mimeType', 'application/octet-stream')
            file_name = file.get('name', 'unnamed')
            
            # Handle Google Workspace files
            if mime_type.startswith('application/vnd.google-apps'):
                return await self._read_workspace_file(file_id, mime_type, file_name)
            
            # Handle PDFs specially
            if mime_type == 'application/pdf' and PDF_SUPPORT:
                return await self._read_pdf_file(file_id, file_name)
            
            # Handle regular files
            return await self._read_regular_file(file_id, mime_type, file_name)
            
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    async def _read_pdf_file(self, file_id: str, file_name: str) -> str:
        """Extract text from PDF files"""
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            fh.seek(0)
            
            # Extract text from PDF
            pdf_reader = pypdf.PdfReader(fh)
            text_content = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {page_num} ---\n{page_text}")
            
            full_text = "\n\n".join(text_content)
            
            return f"Content of {file_name}:\n\n{full_text}"
                
        except Exception as e:
            return f"Failed to extract text from PDF {file_name}: {e}"
    
    async def _read_workspace_file(self, file_id: str, mime_type: str, file_name: str) -> str:
        """Read Google Workspace files (Docs, Sheets, etc.)"""
        export_mime_type = self.EXPORT_FORMATS.get(mime_type, 'text/plain')
        
        try:
            if export_mime_type == 'image/png':
                return f"File {file_name} is a drawing/image and cannot be read as text."
            else:
                # Text export for documents
                response = self.service.files().export(
                    fileId=file_id,
                    mimeType=export_mime_type
                ).execute()
                
                text = response.decode('utf-8') if isinstance(response, bytes) else response
                
                return f"Content of {file_name}:\n\n{text}"
                
        except Exception as e:
            return f"Error reading workspace file {file_name}: {e}"
    
    async def _read_regular_file(self, file_id: str, mime_type: str, file_name: str) -> str:
        """Read regular files from Drive"""
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            content = fh.getvalue()
            
            # Handle HTML files - optionally convert to markdown
            if mime_type == 'text/html' and MARKDOWN_SUPPORT:
                try:
                    html_text = content.decode('utf-8')
                    markdown_text = markdownify(html_text)
                    return f"Content of {file_name} (converted from HTML):\n\n{markdown_text}"
                except:
                    pass  # Fall through to regular text handling
            
            # Return text for text files
            if mime_type.startswith('text/') or mime_type == 'application/json':
                try:
                    text_content = content.decode('utf-8')
                    return f"Content of {file_name}:\n\n{text_content}"
                except UnicodeDecodeError:
                    return f"File {file_name} contains binary data that cannot be displayed as text."
            else:
                return f"File {file_name} is a binary file ({mime_type}) and cannot be displayed as text."
                
        except Exception as e:
            return f"Error reading file {file_name}: {e}"
    
    def start(self, transport: str = "stdio", host: str = None, port: int = None):
        """Start the server"""
        if not GOOGLE_AVAILABLE:
            raise Exception("Google Drive functionality not available - missing dependencies")
        
        try:
            self.initialize_service()
        except Exception as e:
            self.logger.warning(f"Could not initialize Google Drive service: {e}")
            self.logger.info("Server will start but Google Drive functions may not work until GDRIVE_ACCESS_TOKEN is set.")
        
        self.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    server = GoogleDriveMCPServer()
    server.start(transport="stdio")