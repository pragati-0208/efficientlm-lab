"""Tools the agent can call (function calling). Each tool = name + description + JSON schema + python fn.
Nothing here imports torch at module level, so the agent works without a GPU stack."""
import ast
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..summarize import aggregate, load_runs


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments
    fn: Callable[..., str]

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[dict]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, arguments: dict) -> str:
        """Never raises: errors are returned as text so the model can see them and recover."""
        if name not in self._tools:
            return f"ERROR: unknown tool '{name}'. Available: {self.names()}"
        try:
            return str(self._tools[name].fn(**(arguments or {})))
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"


# ------------------------------------------------------------------ calculator
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic by walking the AST - never uses eval(), so no code execution."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / // % ** ( ) are allowed")

    return ev(ast.parse(expression.strip(), mode="eval"))


def calculator(expression: str) -> str:
    value = safe_eval(expression)
    return f"{expression.strip()} = {value:g}" if isinstance(value, float) else f"{expression.strip()} = {value}"


# ------------------------------------------------------------------- sentiment
_POS = {"good", "great", "love", "loved", "excellent", "amazing", "wonderful", "best", "fantastic", "enjoy", "brilliant"}
_NEG = {"bad", "terrible", "hate", "hated", "awful", "worst", "boring", "poor", "horrible", "waste", "disappointing"}


def lexicon_predictor(text: str):
    words = [w.strip(".,!?;:'\"").lower() for w in text.split()]
    score = sum(w in _POS for w in words) - sum(w in _NEG for w in words)
    return ("positive" if score >= 0 else "negative"), 0.5 + min(abs(score), 4) / 10


def load_sentiment_predictor(ckpt: str, model_name: str = "distilbert-base-uncased"):
    """Wrap our fine-tuned classifier as text -> (label, confidence)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    @torch.no_grad()
    def predict(text: str):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=64, return_token_type_ids=False)
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
        i = int(probs.argmax())
        return ("positive" if i == 1 else "negative"), float(probs[i])

    return predict


# --------------------------------------------------------------- experiment info
_METRIC_NAMES = {"test_acc": "test accuracy", "test_f1": "test F1", "latency_ms": "latency (ms/batch)",
                 "trainable_params": "trainable parameters", "model_size_mb": "model size (MB)",
                 "train_time_s": "training time (s)"}
_LOWER_IS_BETTER = {"latency_ms", "trainable_params", "model_size_mb", "train_time_s"}


def make_results_tools(results_dir: str):
    def summarize_results(metric: str = "test_acc") -> str:
        if metric not in _METRIC_NAMES:
            return f"ERROR: metric must be one of {list(_METRIC_NAMES)}"
        agg = aggregate(load_runs(results_dir))
        rows = [(m, e[metric][0], e[metric][1], e["n_seeds"]) for m, e in agg.items() if metric in e]
        if not rows:
            return f"No results found in '{results_dir}'. Run an experiment first."
        rows.sort(key=lambda r: r[1], reverse=metric not in _LOWER_IS_BETTER)
        scale, unit = (100, "%") if metric in ("test_acc",) else (1, "")
        lines = [f"{_METRIC_NAMES[metric]} (best first):"]
        for m, mean, std, n in rows:
            lines.append(f"  {m}: {mean * scale:,.2f}{unit} ± {std * scale:,.2f} (n={n})")
        return "\n".join(lines)

    def get_method_details(method: str) -> str:
        agg = aggregate(load_runs(results_dir))
        if method not in agg:
            return f"ERROR: no results for '{method}'. Known methods: {sorted(agg)}"
        e = agg[method]
        parts = [f"{method} ({e['n_seeds']} seed(s)):"]
        for k, name in _METRIC_NAMES.items():
            if k in e:
                scale = 100 if k == "test_acc" else 1
                parts.append(f"  {name}: {e[k][0] * scale:,.2f} ± {e[k][1] * scale:,.2f}")
        if "adapter_size_kb" in e:
            parts.append(f"  adapter size (KB): {e['adapter_size_kb'][0]:,.0f}")
        return "\n".join(parts)

    return summarize_results, get_method_details


def build_default_registry(results_dir: str = "results", sentiment_predictor: Optional[Callable] = None) -> ToolRegistry:
    predictor = sentiment_predictor or lexicon_predictor
    summarize_results, get_method_details = make_results_tools(results_dir)

    def classify_sentiment(text: str) -> str:
        label, conf = predictor(text)
        return f"sentiment={label} (confidence {conf:.2f})"

    reg = ToolRegistry()
    reg.register(Tool(
        "calculator", "Evaluate an arithmetic expression such as '(12 + 3) * 4'.",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        calculator))
    reg.register(Tool(
        "classify_sentiment", "Classify the sentiment of a piece of text as positive or negative.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        classify_sentiment))
    reg.register(Tool(
        "summarize_results",
        "Rank all experiment methods (full_ft, lora_r8, student_kd, ...) by a metric: "
        + ", ".join(_METRIC_NAMES) + ".",
        {"type": "object", "properties": {"metric": {"type": "string", "enum": list(_METRIC_NAMES)}}},
        summarize_results))
    reg.register(Tool(
        "get_method_details", "Show all measured numbers for one experiment method.",
        {"type": "object", "properties": {"method": {"type": "string"}}, "required": ["method"]},
        get_method_details))
    return reg
