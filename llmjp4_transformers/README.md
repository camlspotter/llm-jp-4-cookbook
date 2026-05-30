# LLM-jp-4 examples for Transformers

`example_tool_call.py` helps inspect whether a model emits Harmony-style tool calls
or falls back to plain JSON / explanation text.
It enables a strict one-shot tool-call example by default; pass `--no-use-one-shot`
to observe the model with fewer formatting hints.
