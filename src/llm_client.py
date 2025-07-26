"""
LLM Client for Response Generation

Handles communication with various LLM providers (OpenAI, Ollama, etc.)
for generating responses in the RAG system.
"""

import logging
import os
from typing import Tuple, Optional, Dict, Any
import openai
impo