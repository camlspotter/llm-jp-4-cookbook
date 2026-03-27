# LLM-jp-4 Cookbook

* Author: Yusuke Oda (@odashi)

This repository contains several examples to use LLM-jp-4 fine-tuned models.

LLM-jp-4 models with the suffix `-base` are basic language models without any fine-tuning.
They are compatible with the base architecture
(Llama for dense models and Qwen for MoE models)
and users are basically able to use these models without special treatment.

LLM-jp-4 models with the suffix `-instruct` are tine-tuned models for chatbot.
They are constructed upon corresponding `-base` models with adopting the
[OpenAI's Harmony Response Format](https://developers.openai.com/cookbook/articles/openai-harmony)
as their default response format.
Harmony brings ability of flexible response construction with reasoning and tool calls,
but users need to apply custom parsing due to lack of better supports for custom tokenizers in the
[official parser](https://github.com/openai/harmony).

At this moment, this repository contains the following subdirectories for specific runtimes:

* [`llmjp4_transformers`](llmjp4_transformers) ... for Huffing Face's [Transformers](https://github.com/huggingface/transformers)
* [`llmjp4_vllm`](llmjp4_vllm) ... for [vLLM](https://github.com/vllm-project/vllm)

All examples are tested upon the following environment:

| | |
|:--- |:--- |
| CPU | Intel Core i9-14900K |
| RAM | 32GiB |
| GPU | NVIDIA RTX 6000 Ada Generation |
| OS | Debian GNU/Linux 12 |
| NVIDIA driver version | 580.119.02 |
| CUDA library version | 12.8 |
