import logging
import os
import re
import uuid
from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
try:
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaFunctionCall,
        DeltaToolCall,
        FunctionCall,
        ToolCall,
    )
except ImportError:
    from vllm.entrypoints.openai.protocol import (
        DeltaFunctionCall,
        DeltaToolCall,
        FunctionCall,
        ToolCall,
    )
try:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
except ImportError:
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
try:
    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
        ExtractedToolCallInformation,
        ToolParser,
        ToolParserManager,
    )
except ImportError:
    from vllm.tool_parsers.abstract_tool_parser import (
        ExtractedToolCallInformation,
        ToolParser,
        ToolParserManager,
    )
from vllm.tokenizers import TokenizerLike

from llmjp4_harmony import HarmonyMessage, HarmonyMessageParser


_RECIPIENT_RE = re.compile(r"(?:^|\s)to=([^\s]+)")
_LOGGER = logging.getLogger(__name__)


@ToolParserManager.register_module(["llmjp4"])
class Llmjp4ToolParser(ToolParser):

    def __init__(self, tokenizer: TokenizerLike, tools=None):
        super().__init__(tokenizer, tools)

        self._parser = HarmonyMessageParser(tokenizer)
        self._assistant_prefill_text = "<|start|>assistant"
        self._assistant_prefill_ids = tokenizer.encode(self._assistant_prefill_text)
        self._start_id = tokenizer.get_vocab()["<|start|>"]
        self._tool_call_ids: list[str] = []
        self._debug_enabled = os.getenv("LLMJP4_VLLM_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        self._debug(
            "adjust_request tool_choice=%r tools=%s skip_special_tokens=%r",
            getattr(request, "tool_choice", None),
            self._get_request_tool_names(request),
            getattr(request, "skip_special_tokens", None),
        )
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        messages = self._parse_messages_from_text(model_output)
        self._debug(
            "extract_tool_calls output_len=%d messages=%s",
            len(model_output),
            self._summarize_messages(messages),
        )

        tool_calls = [
            tool_call
            for message in messages
            if (tool_call := self._build_tool_call(message, request)) is not None
        ]
        content = self._collect_final_content(messages)
        self._debug(
            "extract_tool_calls parsed tool_calls=%s content=%r",
            [self._summarize_tool_call(tool_call) for tool_call in tool_calls],
            content,
        )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=content if content is not None else model_output,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        previous_messages = self._parse_messages_from_ids(previous_token_ids)
        current_messages = self._parse_messages_from_ids(current_token_ids)
        self._debug(
            "extract_tool_calls_streaming prev_tokens=%d curr_tokens=%d prev_messages=%s curr_messages=%s",
            len(previous_token_ids),
            len(current_token_ids),
            self._summarize_messages(previous_messages),
            self._summarize_messages(current_messages),
        )

        content_delta = self._collect_content_delta(previous_messages, current_messages)
        tool_call_delta = self._collect_tool_call_delta(
            previous_messages,
            current_messages,
            request,
        )
        self._debug(
            "extract_tool_calls_streaming content_delta=%r tool_call_delta=%s",
            content_delta,
            [self._summarize_delta_tool_call(delta) for delta in tool_call_delta],
        )

        if content_delta is None and not tool_call_delta:
            return None

        return DeltaMessage(content=content_delta, tool_calls=tool_call_delta or [])

    def _parse_messages_from_text(self, model_output: str) -> list[HarmonyMessage]:
        token_ids = self.model_tokenizer.encode(self._assistant_prefill_text + model_output)
        return self._parser.get_all_messages(token_ids)

    def _parse_messages_from_ids(self, token_ids: Sequence[int]) -> list[HarmonyMessage]:
        token_ids = list(token_ids)
        if not token_ids or token_ids[0] != self._start_id:
            token_ids = self._assistant_prefill_ids + token_ids
        return self._parser.get_all_messages(token_ids)

    def _collect_final_content(self, messages: Sequence[HarmonyMessage]) -> str | None:
        parts: list[str] = []

        for message in messages:
            if not self._is_final_message(message):
                continue
            parts.append(self.model_tokenizer.decode(message.content.token_ids))

        if not parts:
            return None
        return "".join(parts)

    def _collect_content_delta(
        self,
        previous_messages: Sequence[HarmonyMessage],
        current_messages: Sequence[HarmonyMessage],
    ) -> str | None:
        previous_content = self._collect_final_content(previous_messages) or ""
        current_content = self._collect_final_content(current_messages) or ""

        if not current_content or current_content == previous_content:
            return None
        if current_content.startswith(previous_content):
            return current_content[len(previous_content):] or None
        return current_content

    def _collect_tool_call_delta(
        self,
        previous_messages: Sequence[HarmonyMessage],
        current_messages: Sequence[HarmonyMessage],
        request: ChatCompletionRequest,
    ) -> list[DeltaToolCall]:
        previous_calls = self._extract_tool_call_states(previous_messages, request)
        current_calls = self._extract_tool_call_states(current_messages, request)

        deltas: list[DeltaToolCall] = []

        for index, (name, arguments) in enumerate(current_calls):
            while len(self._tool_call_ids) <= index:
                self._tool_call_ids.append(f"call_{uuid.uuid4().hex}")

            if index >= len(previous_calls):
                deltas.append(
                    DeltaToolCall(
                        index=index,
                        id=self._tool_call_ids[index],
                        type="function",
                        function=DeltaFunctionCall(
                            name=name,
                            arguments=arguments,
                        ).model_dump(exclude_none=True),
                    )
                )
                continue

            previous_name, previous_arguments = previous_calls[index]
            function_delta: dict[str, str] = {}

            if name != previous_name:
                function_delta["name"] = name

            if arguments != previous_arguments:
                if previous_arguments and arguments.startswith(previous_arguments):
                    argument_delta = arguments[len(previous_arguments):]
                else:
                    argument_delta = arguments
                if argument_delta:
                    function_delta["arguments"] = argument_delta

            if function_delta:
                deltas.append(
                    DeltaToolCall(
                        index=index,
                        id=self._tool_call_ids[index],
                        type="function",
                        function=DeltaFunctionCall(**function_delta).model_dump(
                            exclude_none=True
                        ),
                    )
                )

        return deltas

    def _extract_tool_call_states(
        self,
        messages: Sequence[HarmonyMessage],
        request: ChatCompletionRequest,
    ) -> list[tuple[str, str]]:
        states: list[tuple[str, str]] = []

        for message in messages:
            if not self._is_tool_call_message(message):
                continue

            recipient = self._extract_recipient(message)
            if recipient is None:
                continue

            name = self._normalize_tool_name(recipient, request)
            arguments = self._extract_arguments_text(message)
            states.append((name, arguments))

        self._debug("tool_call_states=%s", states)
        return states

    def _build_tool_call(
        self,
        message: HarmonyMessage,
        request: ChatCompletionRequest,
    ) -> ToolCall | None:
        if not self._is_tool_call_message(message):
            return None

        recipient = self._extract_recipient(message)
        if recipient is None:
            return None

        tool_call = ToolCall(
            type="function",
            function=FunctionCall(
                name=self._normalize_tool_name(recipient, request),
                arguments=self._extract_arguments_text(message),
            ),
        )
        self._debug(
            "build_tool_call recipient=%r tool_call=%s",
            recipient,
            self._summarize_tool_call(tool_call),
        )
        return tool_call

    def _is_final_message(self, message: HarmonyMessage) -> bool:
        if message.role is None or message.channel is None or message.content is None:
            return False
        role_text = self.model_tokenizer.decode(message.role.token_ids).strip()
        channel_text = self.model_tokenizer.decode(message.channel.token_ids).strip()
        channel_name = channel_text.split(maxsplit=1)[0] if channel_text else ""
        return role_text == "assistant" and channel_name == "final"

    def _is_tool_call_message(self, message: HarmonyMessage) -> bool:
        if message.role is None or message.channel is None or message.content is None:
            return False

        role_text = self.model_tokenizer.decode(message.role.token_ids).strip()
        if role_text != "assistant":
            return False

        channel_text = self.model_tokenizer.decode(message.channel.token_ids).strip()
        channel_name = channel_text.split(maxsplit=1)[0] if channel_text else ""
        if channel_name != "commentary":
            return False

        return self._extract_recipient(message) is not None

    def _extract_recipient(self, message: HarmonyMessage) -> str | None:
        headers = []
        if message.role is not None:
            headers.append(self.model_tokenizer.decode(message.role.token_ids))
        if message.channel is not None:
            headers.append(self.model_tokenizer.decode(message.channel.token_ids))

        for header in headers:
            match = _RECIPIENT_RE.search(header)
            if match:
                return match.group(1)
        return None

    def _normalize_tool_name(
        self,
        recipient: str,
        request: ChatCompletionRequest,
    ) -> str:
        candidate = recipient.rsplit(".", maxsplit=1)[-1]

        request_tools = getattr(request, "tools", None) or []
        known_names: set[str] = set()
        for tool in request_tools:
            function = getattr(tool, "function", None)
            name = getattr(function, "name", None)
            if name:
                known_names.add(name)

        if recipient in known_names:
            return recipient
        if candidate in known_names:
            return candidate
        return candidate

    def _extract_arguments_text(self, message: HarmonyMessage) -> str:
        if message.content is None:
            return ""
        return self.model_tokenizer.decode(message.content.token_ids).strip()

    def _debug(self, message: str, *args) -> None:
        if self._debug_enabled and _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("[llmjp4_tool_parser] " + message, *args)

    def _get_request_tool_names(self, request: ChatCompletionRequest) -> list[str]:
        tool_names: list[str] = []
        for tool in getattr(request, "tools", None) or []:
            function = getattr(tool, "function", None)
            name = getattr(function, "name", None)
            if name:
                tool_names.append(name)
        return tool_names

    def _summarize_messages(
        self,
        messages: Sequence[HarmonyMessage],
    ) -> list[dict[str, str | int | None]]:
        summarized: list[dict[str, str | int | None]] = []
        for message in messages:
            role = (
                self.model_tokenizer.decode(message.role.token_ids)
                if message.role is not None
                else None
            )
            channel = (
                self.model_tokenizer.decode(message.channel.token_ids)
                if message.channel is not None
                else None
            )
            constrain = (
                self.model_tokenizer.decode(message.constrain.token_ids)
                if message.constrain is not None
                else None
            )
            content = (
                self.model_tokenizer.decode(message.content.token_ids)
                if message.content is not None
                else None
            )
            summarized.append(
                {
                    "end": message.end.name,
                    "role": role,
                    "channel": channel,
                    "constrain": constrain,
                    "content": content,
                    "content_len": len(content) if content is not None else None,
                }
            )
        return summarized

    def _summarize_tool_call(self, tool_call: ToolCall) -> dict[str, str | None]:
        return {
            "type": getattr(tool_call, "type", None),
            "name": getattr(tool_call.function, "name", None),
            "arguments": getattr(tool_call.function, "arguments", None),
        }

    def _summarize_delta_tool_call(
        self,
        tool_call: DeltaToolCall,
    ) -> dict[str, str | int | None | dict]:
        return {
            "index": getattr(tool_call, "index", None),
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", None),
            "function": getattr(tool_call, "function", None),
        }
