from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "llm-jp/llm-jp-4-8b-thinking",
    trust_remote_code=True,
)
print(tok.chat_template)
