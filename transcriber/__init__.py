"""
Transcriber - AI Voice Agent

A voice interface for interacting with an AI agent capable of executing tools.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

try:
    from .main import app
    __all__ = ["app"]
except ImportError:
    # Handle missing dependencies gracefully
    __all__ = []