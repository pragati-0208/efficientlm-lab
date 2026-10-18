"""'Brains' for the agent. Both implement  generate(history, tool_schemas) -> LLMResponse.

history is a provider-neutral list of dicts:
  {"role": "user", "content": str}
  {"role": "tool_call", "id": str, "name": str, "arguments": dict}
  {"role": "tool_result", "id": str, "name": str, "content": str}

RuleBasedLLM : offline, deterministic, no API key -> good for tests and demos.
AnthropicLLM : real LLM function calling through the Anthropic API (needs ANTHROPIC_API_KEY).
"""
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMResponse:
    kind: str  # "tool_call" or "final"
    content: str = ""
    name: Optional[str] = None
    arguments: dict = field(default_factory=dict)
    call_id: Optional[str] = None
    thought: str = ""


KNOWN_METHODS = ["full_ft", "lora_peft_r8", "lora_r4", "lora_r8", "lora_r16", "student_ce", "student_kd"]


class RuleBasedLLM:
    """A tiny intent router that *behaves* like a function-calling model: it picks a tool, fills
    the arguments, then writes the final answer from the tool result."""

    def generate(self, history: List[dict], tools: List[dict]) -> LLMResponse:
        last = history[-1]
        if last["role"] == "tool_result":
            return LLMResponse("final", content=last["content"])
        text = last["content"]
        low = text.lower()

        quoted = re.search(r"[\"“'‘](.+?)[\"”'’]", text)
        if quoted and re.search(r"sentiment|positive|negative|feel|opinion", low):
            return LLMResponse("tool_call", name="classify_sentiment", arguments={"text": quoted.group(1)},
                               thought="The user wants the sentiment of a quoted text -> classify_sentiment.")

        if re.search(r"best|accuracy|compare|results|experiment|latency|params|parameters|size|f1|fastest|smallest|"
                     r"distill|lora|full_ft|student", low):
            for m in KNOWN_METHODS:
                if m in low:
                    return LLMResponse("tool_call", name="get_method_details", arguments={"method": m},
                                       thought=f"Question about one method ({m}) -> get_method_details.")
            metric = "test_acc"
            if "latency" in low or "fastest" in low or "speed" in low:
                metric = "latency_ms"
            elif "param" in low or "smallest" in low:
                metric = "trainable_params"
            elif "f1" in low:
                metric = "test_f1"
            elif "size" in low:
                metric = "model_size_mb"
            return LLMResponse("tool_call", name="summarize_results", arguments={"metric": metric},
                               thought=f"Question about experiment results ({metric}) -> summarize_results.")

        expr = re.search(r"[\d.\s+\-*/()%^]{3,}", text)
        if expr and re.search(r"\d", expr.group()) and re.search(r"[+\-*/%^]", expr.group()):
            return LLMResponse("tool_call", name="calculator",
                               arguments={"expression": expr.group().strip().replace("^", "**")},
                               thought="The user wants arithmetic -> calculator.")

        return LLMResponse("final", content=(
            "I can: (1) do arithmetic, (2) classify the sentiment of a quoted sentence, "
            "(3) report experiment results (e.g. 'which method has the best accuracy?')."))


class AnthropicLLM:
    SYSTEM = ("You are an assistant for an ML mini-project. Use the provided tools whenever they can answer "
              "the question; never guess numbers that a tool can provide. Be concise.")

    def __init__(self, model: Optional[str] = None):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    @staticmethod
    def to_messages(history: List[dict]) -> List[dict]:
        msgs: List[dict] = []
        for h in history:
            if h["role"] == "user":
                msgs.append({"role": "user", "content": h["content"]})
            elif h["role"] == "tool_call":
                block = {"type": "tool_use", "id": h["id"], "name": h["name"], "input": h["arguments"]}
                if msgs and msgs[-1]["role"] == "assistant" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "assistant", "content": [block]})
            elif h["role"] == "tool_result":
                block = {"type": "tool_result", "tool_use_id": h["id"], "content": h["content"]}
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "user", "content": [block]})
        return msgs

    def generate(self, history: List[dict], tools: List[dict]) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, system=self.SYSTEM, tools=tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=self.to_messages(history),
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        for b in resp.content:
            if b.type == "tool_use":
                return LLMResponse("tool_call", name=b.name, arguments=dict(b.input), call_id=b.id, thought=text)
        return LLMResponse("final", content=text)
