"""The agent loop: ask the model -> if it wants a tool, run it, feed the result back -> repeat."""
from dataclasses import dataclass, field
from typing import List

from .tools import ToolRegistry


@dataclass
class AgentResult:
    answer: str
    trace: List[dict] = field(default_factory=list)  # every thought / tool call / observation
    steps: int = 0


class Agent:
    def __init__(self, llm, registry: ToolRegistry, max_steps: int = 5, verbose: bool = False):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps  # hard stop: an agent must never loop forever
        self.verbose = verbose

    def run(self, question: str) -> AgentResult:
        history = [{"role": "user", "content": question}]
        trace: List[dict] = []
        for step in range(1, self.max_steps + 1):
            resp = self.llm.generate(history, self.registry.schemas())
            if resp.thought:
                trace.append({"type": "thought", "content": resp.thought})
                self._log(f"thought: {resp.thought}")
            if resp.kind == "final":
                trace.append({"type": "final", "content": resp.content})
                return AgentResult(resp.content, trace, step)

            call_id = resp.call_id or f"call_{step}"
            history.append({"role": "tool_call", "id": call_id, "name": resp.name, "arguments": resp.arguments})
            trace.append({"type": "tool_call", "name": resp.name, "arguments": resp.arguments})
            self._log(f"call: {resp.name}({resp.arguments})")
            result = self.registry.call(resp.name, resp.arguments)
            history.append({"role": "tool_result", "id": call_id, "name": resp.name, "content": result})
            trace.append({"type": "observation", "content": result})
            self._log(f"observation: {result}")
        return AgentResult(f"Stopped after {self.max_steps} steps without a final answer.", trace, self.max_steps)

    def _log(self, msg):
        if self.verbose:
            print(f"  [agent] {msg}")
