# This script works similarly to the `vllm` CLI command,
# but registers additional components.
#
# Usage: python example_cli.py [...rest of vllm CLI arguments]
#
# Example:
# The following command runs the llm-jp-4-8b-thinking model with the llmjp4
# reasoning parser and tool call parser.
# python example_cli.py serve llm-jp/llm-jp-4-8b-thinking --reasoning-parser llmjp4 --tool-call-parser llmjp4 --enable-auto-tool-choice --trust-remote-code

import logging
import os

from vllm.entrypoints.cli import main as cli_main

# Load the custom reasoning before launching the CLI.
import llmjp4_reasoning_parser
import llmjp4_tool_parser


if __name__ == "__main__":
    if os.getenv("LLMJP4_VLLM_DEBUG", "").lower() in {"1", "true", "yes", "on"}:
        logger = logging.getLogger("llmjp4_tool_parser")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            logger.addHandler(handler)
        logger.propagate = False
    cli_main.main()
