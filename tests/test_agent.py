import pytest

from efficientlm.agent import Agent, build_default_registry
from efficientlm.agent.llm import AnthropicLLM, RuleBasedLLM
from efficientlm.agent.tools import safe_eval


def make_agent(results_dir, max_steps=5):
    return Agent(RuleBasedLLM(), build_default_registry(results_dir), max_steps=max_steps)


def test_safe_eval_arithmetic():
    assert safe_eval("12 * (3 + 4)") == 84
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("-5 + 2") == -3


@pytest.mark.parametrize("bad", ["__import__('os').system('echo hi')", "open('x')", "1 + a", "2 ** 1000"])
def test_safe_eval_rejects_code(bad):
    with pytest.raises(Exception):
        safe_eval(bad)


def test_agent_calculator(fake_results):
    res = make_agent(fake_results).run("What is 12 * (3 + 4)?")
    assert "84" in res.answer
    assert [t["name"] for t in res.trace if t["type"] == "tool_call"] == ["calculator"]


def test_agent_sentiment(fake_results):
    res = make_agent(fake_results).run("Is 'I loved this great movie' positive or negative?")
    assert "positive" in res.answer
    res = make_agent(fake_results).run("What is the sentiment of 'this was a terrible, boring waste'?")
    assert "negative" in res.answer


def test_agent_reads_experiment_results(fake_results):
    res = make_agent(fake_results).run("Which method has the best accuracy?")
    lines = [l.strip() for l in res.answer.splitlines()]
    assert lines[1].startswith("full_ft")  # ranked best-first


def test_agent_latency_ranking_is_lower_is_better(fake_results):
    res = make_agent(fake_results).run("Which method has the lowest latency?")
    assert [l.strip() for l in res.answer.splitlines()][1].startswith("student_kd")


def test_agent_method_details(fake_results):
    res = make_agent(fake_results).run("Tell me the accuracy of lora_r8")
    assert "lora_r8" in res.answer and "test accuracy" in res.answer


def test_agent_unknown_question_gives_help(fake_results):
    res = make_agent(fake_results).run("Tell me a joke")
    assert "arithmetic" in res.answer


def test_agent_max_steps_guard(fake_results):
    class Looping:
        def generate(self, history, tools):
            from efficientlm.agent.llm import LLMResponse
            return LLMResponse("tool_call", name="calculator", arguments={"expression": "1+1"})

    res = Agent(Looping(), build_default_registry(fake_results), max_steps=3).run("x")
    assert res.steps == 3 and "Stopped" in res.answer


def test_registry_returns_errors_instead_of_raising(fake_results):
    reg = build_default_registry(fake_results)
    assert reg.call("nope", {}).startswith("ERROR")
    assert reg.call("calculator", {}).startswith("ERROR")
    assert reg.call("calculator", {"expression": "1/0"}).startswith("ERROR")


def test_anthropic_message_conversion():
    history = [
        {"role": "user", "content": "q"},
        {"role": "tool_call", "id": "a", "name": "calculator", "arguments": {"expression": "1+1"}},
        {"role": "tool_result", "id": "a", "name": "calculator", "content": "1+1 = 2"},
    ]
    msgs = AnthropicLLM.to_messages(history)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[2]["content"][0]["tool_use_id"] == "a"
