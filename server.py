#!/usr/bin/env python3
import json
import sys
import logging
import urllib.request
import urllib.error
import re
import time
from typing import Dict, List, Any, Set, Optional
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from ollama import chat, list as ollama_list

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

server = Server("local-think-mcp")

# Cache for thinking models
THINKING_MODELS_CACHE: Optional[Set[str]] = None
CACHE_TIMESTAMP = 0
CACHE_DURATION = 3600  # 1 hour in seconds

def fetch_thinking_models() -> Set[str]:
    """Fetch the list of thinking models from Ollama website."""
    global THINKING_MODELS_CACHE, CACHE_TIMESTAMP
    
    # Return cached data if still valid
    if THINKING_MODELS_CACHE is not None and time.time() - CACHE_TIMESTAMP < CACHE_DURATION:
        return THINKING_MODELS_CACHE
    
    try:
        # Fetch the HTML content
        req = urllib.request.Request(
            'https://ollama.com/search?c=thinking',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
        
        # Parse model names from the HTML
        # Look for patterns like href="/library/model-name"
        model_pattern = r'href="/library/([^"]+)"'
        matches = re.findall(model_pattern, html)
        
        # Filter out duplicates and clean model names
        thinking_models = set()
        for match in matches:
            # Extract base model name (without version)
            model_name = match.split('/')[0].lower()
            thinking_models.add(model_name)
        
        # Add known thinking models as fallback
        thinking_models.update(['deepseek-r1', 'qwen3', 'deepseek-rl', 'qwq'])
        
        # Update cache
        THINKING_MODELS_CACHE = thinking_models
        CACHE_TIMESTAMP = time.time()
        
        logger.info(f"Fetched thinking models: {thinking_models}")
        return thinking_models
        
    except Exception as e:
        logger.warning(f"Failed to fetch thinking models from website: {e}")
        # Return default known thinking models on error
        default_models = {'deepseek-r1', 'qwen3', 'deepseek-rl', 'qwq'}
        THINKING_MODELS_CACHE = default_models
        CACHE_TIMESTAMP = time.time()
        return default_models

def is_thinking_model(model_name: str) -> bool:
    """Check if a model supports thinking tokens."""
    thinking_models = fetch_thinking_models()
    model_lower = model_name.lower()
    
    # Check if any thinking model name is in the model name
    for thinking_model in thinking_models:
        if thinking_model in model_lower:
            return True
    return False

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="chat_with_thinking",
            description="Chat with an Ollama model and get only the thinking tokens (internal reasoning)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "The Ollama model to use (e.g., 'deepseek-rl:14b')"
                    },
                    "prompt": {
                        "type": "string", 
                        "description": "The prompt to send to the model"
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system prompt",
                        "default": ""
                    }
                },
                "required": ["model", "prompt"]
            }
        ),
        types.Tool(
            name="list_models",
            description="List available Ollama models that support thinking tokens",
            inputSchema={
                "type": "object",
                "properties": {
                    "all_models": {
                        "type": "boolean",
                        "description": "If true, list all models. If false (default), only list models with thinking support",
                        "default": False
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    if name == "chat_with_thinking":
        model = arguments.get("model")
        prompt = arguments.get("prompt")
        system = arguments.get("system", "")
        
        if not model or not prompt:
            return [types.TextContent(
                type="text",
                text="Error: 'model' and 'prompt' are required parameters"
            )]
        
        # Check if the model supports thinking
        if not is_thinking_model(model):
            # Get list of installed thinking models
            models = ollama_list()
            installed_thinking_models = [
                m.model for m in models.models 
                if is_thinking_model(m.model)
            ]
            
            error_msg = f"Error: Model '{model}' does not support thinking tokens.\n"
            if installed_thinking_models:
                error_msg += f"Available thinking models: {', '.join(installed_thinking_models)}"
            else:
                error_msg += "No thinking models are installed. Try: ollama pull deepseek-r1"
            
            return [types.TextContent(
                type="text",
                text=error_msg
            )]
        
        try:
            messages = []
            if system:
                messages.append({
                    'role': 'system',
                    'content': system
                })
            messages.append({
                'role': 'user',
                'content': prompt
            })
            
            response = chat(model, messages=messages, think=True)
            
            result = {
                "thinking": response.message.thinking if hasattr(response.message, 'thinking') else "",
                "model": model
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error calling Ollama: {str(e)}"
            )]
    
    elif name == "list_models":
        try:
            all_models = arguments.get("all_models", False)
            models = ollama_list()
            model_list = []
            
            for model in models.models:
                model_name = model.model
                
                # Check if this model supports thinking using dynamic detection
                supports_thinking = is_thinking_model(model_name)
                
                # Include model if showing all or if it supports thinking
                if all_models or supports_thinking:
                    model_info = {
                        "name": model.model,
                        "size": f"{model.size / (1024**3):.1f}GB" if hasattr(model, 'size') and model.size else "Unknown",
                        "modified": str(model.modified_at) if hasattr(model, 'modified_at') and model.modified_at else "Unknown",
                        "supports_thinking": supports_thinking
                    }
                    model_list.append(model_info)
            
            response = {
                "models": model_list,
                "filter": "all models" if all_models else "thinking models only"
            }
            
            if not all_models and not model_list:
                response["note"] = "No thinking models found. Install one with: ollama pull deepseek-rl:14b"
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error listing models: {str(e)}"
            )]
    
    else:
        return [types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="local-think-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())