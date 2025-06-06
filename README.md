# Local Think MCP

An MCP (Model Context Protocol) server that interfaces with Ollama to get both thinking tokens and response content from language models.

## Features

- Chat with any Ollama model and retrieve thinking tokens
- List all available Ollama models
- Dynamic model selection
- Support for system prompts

## Prerequisites

- Python 3.8+
- Ollama installed and running
- Models that support thinking tokens (e.g., deepseek-rl:14b)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running:
```bash
ollama serve
```

3. Pull a model that supports thinking tokens:
```bash
ollama pull deepseek-rl:14b
```

## Usage

### As an MCP Server

Add this to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "local-think": {
      "command": "python",
      "args": ["/path/to/local_think/server.py"]
    }
  }
}
```

### Tools Available

1. **chat_with_thinking**
   - Send a prompt to an Ollama model
   - Returns both thinking tokens and final response
   - Parameters:
     - `model` (required): The Ollama model to use
     - `prompt` (required): Your prompt
     - `system` (optional): System prompt

2. **list_models**
   - Lists Ollama models that support thinking tokens (by default)
   - Parameters:
     - `all_models` (optional): Set to true to list all models, not just thinking models

### Example Usage

```python
# Using chat_with_thinking
{
  "model": "deepseek-rl:14b",
  "prompt": "What is 10 + 23?",
  "system": "You are a helpful assistant."
}

# Response format:
{
  "thinking": "Let me calculate 10 + 23...",
  "content": "10 + 23 = 33",
  "model": "deepseek-rl:14b"
}
```

## Notes

- Not all Ollama models support thinking tokens
- The `think=True` parameter requires models specifically trained with thinking token support
- If a model doesn't support thinking tokens, the thinking field will be empty
- Known thinking models include: deepseek-rl, deepseek-r1, qwq, and models with "thinking" in their name