# Example script to use LLM-jp-4 models with vLLM.

import argparse

from vllm import LLM, SamplingParams

from llmjp4_harmony import HarmonyMessageParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        # trust_remote_code is required to load custom tokenizer and reasoning parser.
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    messages = [
        {"role": "user", "content": "日本語で自己紹介してください。"},
    ]

    prompt: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("--- Prompt ---")
    print(prompt)

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )

    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0].outputs[0]

    print("--- Generated IDs ---")
    print(output.token_ids)

    # NOTE(odashi):
    # Don't use `output.text` at this moment.
    # It doesn't handle whitespaces appropriately.
    decoded_output = tokenizer.decode(output.token_ids)
    print("\n--- Decoded Output ---")
    print(decoded_output)

    parser = HarmonyMessageParser(tokenizer)
    print("\n--- Parsed Harmony Messages ---")
    for i, message in enumerate(parser.iter_messages(output.token_ids), start=1):
        print(f"Message {i}:")

        # The end type can be "END", "CALL", or "INCOMPLETE".
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
            print("  Constrain Text:", repr(tokenizer.decode(message.constrain.token_ids)))
            print("  Constrain Start Position:", message.constrain.start)
        if message.content:
            print("  Content Tokens:", message.content.token_ids)
            print("  Content Text:", repr(tokenizer.decode(message.content.token_ids)))
            print("  Content Start Position:", message.content.start)


if __name__ == "__main__":
    main()
