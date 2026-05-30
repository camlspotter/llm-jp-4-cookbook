# LLM-jp-4 examples for vLLM

Use `example_cli.py` with both `--reasoning-parser llmjp4` and
`--tool-call-parser llmjp4` when serving Harmony-based tool calling models.

Set `LLMJP4_VLLM_DEBUG=1` and raise the Python logging level to `DEBUG` to inspect
how the custom tool parser interprets Harmony messages inside vLLM.
