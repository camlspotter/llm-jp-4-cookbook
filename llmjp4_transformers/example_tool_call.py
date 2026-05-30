import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect tool-calling behavior of LLM-jp-4 with Transformers."
    )
    parser.add_argument(
        "--model",
        default="llm-jp/llm-jp-4-8b-thinking",
        help="Model name on Hugging Face Hub.",
    )
    parser.add_argument(
        "--prompt",
        default="東京の現在の天気を取得してください。必ず get_weather を使ってください。",
        help="User prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort passed to the chat template.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt. If omitted, no system message is added.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    tools = build_tools()
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})

    prompt: str = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=args.reasoning_effort,
    )

    print("--- Tools ---")
    print(tools)

    print("\n--- Prompt ---")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\n--- Input IDs ---")
    print(inputs["input_ids"][0].tolist())

    with torch.no_grad():
        output_tensor = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    generated_ids: list[int] = output_tensor[0][inputs["input_ids"].shape[1]:].tolist()

    print("\n--- Generated IDs ---")
    print(generated_ids)

    response = tokenizer.decode(generated_ids)

    print("\n--- Response ---")
    print(response)

    parsed = tokenizer.parse_response(response)

    print("\n--- Parsed Response ---")
    print("Role:", parsed.get("role"))
    print("Thinking:", parsed.get("thinking"))
    print("Content:", parsed.get("content"))

    response_prefill = tokenizer.encode("<|start|>assistant")
    parsed_harmony = tokenizer.parse_harmony_message(response_prefill + generated_ids)

    print("\n--- Parsed Harmony Messages ---")
    for i, message in enumerate(parsed_harmony, start=1):
        print(f"Message {i}:")
        print("  End Type:", message.end)

        if message.role:
            print("  Role Tokens:", message.role.token_ids)
            print("  Role Text:", repr(tokenizer.decode(message.role.token_ids)))
            print("  Role Start Position:", message.role.start)
        if message.channel:
            print("  Channel Tokens:", message.channel.token_ids)
            print("  Channel Text:", repr(tokenizer.decode(message.channel.token_ids)))
            print("  Channel Start Position:", message.channel.start)
        if message.constrain:
            print("  Constrain Tokens:", message.constrain.token_ids)
            print(
                "  Constrain Text:",
                repr(tokenizer.decode(message.constrain.token_ids)),
            )
            print("  Constrain Start Position:", message.constrain.start)
        if message.content:
            print("  Content Tokens:", message.content.token_ids)
            print("  Content Text:", repr(tokenizer.decode(message.content.token_ids)))
            print("  Content Start Position:", message.content.start)


if __name__ == "__main__":
    main()
