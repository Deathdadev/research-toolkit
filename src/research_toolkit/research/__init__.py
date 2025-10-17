"""
Research tools and MCP server for AI integration.

This module provides the Model Context Protocol (MCP) server and research
methodology tools for AI model integration.

Version 2.0.0 features:
- 8 research types
- 10 APA 7 reference types
- Advanced name parsing
"""

from .mcp_server import ResearchToolkitMCPServer, EmpiricalResearchMCPServer

__all__ = [
    'ResearchToolkitMCPServer',
    'EmpiricalResearchMCPServer'  # Backward compatibility
]
