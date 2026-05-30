import argparse
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from example_tool_call import build_one_shot_instruction, build_tools


class InspectRequest(BaseModel):
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    system_prompt: str | None = None
    use_one_shot: bool = True
    reasoning_effort: str = "medium"
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


class AppState:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()


def build_default_messages(
    system_prompt: str | None,
    use_one_shot: bool,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_parts: list[str] = []
    if system_prompt:
        system_parts.append(system_prompt)
    if use_one_shot:
        system_parts.append(build_one_shot_instruction())
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.append(
        {
            "role": "user",
            "content": "東京の現在の天気を取得してください。必ず get_weather を使ってください。",
        }
    )
    return messages


def serialize_harmony_message(tokenizer, message) -> dict[str, Any]:
    payload: dict[str, Any] = {"end": message.end.name}
    for key in ("role", "channel", "constrain", "content"):
        value = getattr(message, key)
        if value is None:
            payload[key] = None
            continue
        payload[key] = {
            "token_ids": value.token_ids,
            "text": tokenizer.decode(value.token_ids),
            "start": value.start,
        }
    return payload


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="LLM-jp-4 Transformers Example Server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": state.model_name}

    @app.post("/inspect")
    def inspect(request: InspectRequest) -> dict[str, Any]:
        messages = request.messages or build_default_messages(
            system_prompt=request.system_prompt,
            use_one_shot=request.use_one_shot,
        )
        tools = request.tools if request.tools is not None else build_tools()

        prompt: str = state.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=request.reasoning_effort,
        )

        inputs = state.tokenizer(prompt, return_tensors="pt").to(state.model.device)

        with torch.no_grad():
            output_tensor = state.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
            )

        generated_ids: list[int] = output_tensor[0][
            inputs["input_ids"].shape[1]:
        ].tolist()
        response = state.tokenizer.decode(generated_ids)
        parsed = state.tokenizer.parse_response(response)
        response_prefill = state.tokenizer.encode("<|start|>assistant")
        parsed_harmony = state.tokenizer.parse_harmony_message(
            response_prefill + generated_ids
        )

        return {
            "model": state.model_name,
            "messages": messages,
            "tools": tools,
            "prompt": prompt,
            "input_ids": inputs["input_ids"][0].tolist(),
            "generated_ids": generated_ids,
            "response": response,
            "parsed_response": {
                "role": parsed.get("role"),
                "thinking": parsed.get("thinking"),
                "content": parsed.get("content"),
            },
            "harmony_messages": [
                serialize_harmony_message(state.tokenizer, message)
                for message in parsed_harmony
            ],
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve LLM-jp-4 Transformers examples without reloading the model."
    )
    parser.add_argument(
        "--model",
        default="llm-jp/llm-jp-4-8b-thinking",
        help="Model name on Hugging Face Hub.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=18080, help="Port to bind.")
    return parser.parse_args()


def main():
    args = parse_args()
    state = AppState(args.model)
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
