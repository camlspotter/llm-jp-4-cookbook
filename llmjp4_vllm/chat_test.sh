#!/bin/bash

# Example script to communicate with the vLLM server using curl.
# Before running this script, make sure to start the vLLM server with LLM-jp-4 models loaded.
# You can use the example_cli.py script in this directory to start the server.

# "stream": true requests streaming mode, which periodically returns partial responses.
# This mode returns both the reasoning and the final response.

# "stream": false requests non-streaming mode, which returns only the final response.
# This is due to the limitation of the current vLLM interface.

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm-jp/llm-jp-4-8b-thinking",
    "messages": [{"role": "user", "content": "二次方程式の解の公式を導出して下さい。"}],
    "stream": true
  }'
