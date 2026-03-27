# LLM-jp-4 Cookbook

* Author: Yusuke Oda (@odashi)

## Notes for `llm-jp-4-*-instruct` Models

This repository contains several examples to use LLM-jp-4 fine-tuned models.

LLM-jp-4 models with the suffix `-instruct` are fine-tuned models for chatbot applications.
They are constructed upon corresponding `-base` models in the same model series,
with adopting the
[OpenAI's Harmony Response Format](https://developers.openai.com/cookbook/articles/openai-harmony)
as their default response structure.

Harmony brings ability of flexible response construction with reasoning and tool calls,
but users need to apply custom parsing due to lack of fine-grained supports for custom tokenizers in the
[official parser implementation](https://github.com/openai/harmony).

Specifically, users need to take care about:

* **Tokenizer**: LLM-jp-4 models are using LlamaTokenizer (Sentencepiece),
  but users need to take additional care before detokenizing output tokens
  into the resulting text to avoid known issues around the Sentencepiece library
  [(1)](https://github.com/huggingface/transformers/pull/26678)
  [(2)](https://github.com/huggingface/transformers/issues/28218).
* **Input Template**: Users need to apply Harmony to their chat inputs.
  This is basically achieved by using the bundled chat template (jinja2) in the LLM-jp-4 models,
  but for some cases users might need to implement their own encoding strategy
  (e.g., the case that user inputs contain the same string with special tokens `<|...|>`).
* **Output Parsing**: Since Harmony is a token-based encoding,
  users might need to analyse output tokens directly rather than detokenized texts
  to obtain accurate parsing results (due to the same reason described in the "Input Templates").
  For convenience, LLM-jp-4 models also bundle a parser library for Harmony-encoded tokens
  (`llmjp4_harmony.py`).

At this moment, this repository contains the following subdirectories for specific LLM runtimes:

* [`llmjp4_transformers`](llmjp4_transformers) ... for Huffing Face's [Transformers](https://github.com/huggingface/transformers)
* [`llmjp4_vllm`](llmjp4_vllm) ... for [vLLM](https://github.com/vllm-project/vllm)

## Notes for Other `llm-jp-4` Models

LLM-jp-4 models with the suffix `-base` are basic language models without any fine-tuning.
Their behavior is basically compatible with the base architecture
(Llama for dense models and Qwen for MoE models)
and users are able to use these models without special treatment.

Note that if users are trying to use some special tokens in the `-base` models or their inheritances,
or adding their own special tokens into the vocabulary,
users may encounter the same issues described above.
To provide the same solution, `-base` models also bundle the same functionality
with `-instruct` models.

## Test Environments

All examples are tested using the following environment:

| | |
|:--- |:--- |
| CPU | Intel Core i9-14900K |
| RAM | 32GiB |
| GPU | NVIDIA RTX 6000 Ada Generation |
| OS | Debian GNU/Linux 12 |
| NVIDIA driver version | 580.119.02 |
| CUDA library version | 12.8 |
