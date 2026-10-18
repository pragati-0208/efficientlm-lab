"""Interactive demo:  python -m efficientlm.agent.cli --llm rule   (or --llm anthropic)"""
import argparse

from .agent import Agent
from .llm import AnthropicLLM, RuleBasedLLM
from .tools import build_default_registry, load_sentiment_predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["rule", "anthropic"], default="rule")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--ckpt", default=None, help="fine-tuned classifier state_dict (else a keyword fallback is used)")
    ap.add_argument("--question", default=None, help="ask one question and exit")
    args = ap.parse_args()

    predictor = load_sentiment_predictor(args.ckpt) if args.ckpt else None
    llm = RuleBasedLLM() if args.llm == "rule" else AnthropicLLM()
    agent = Agent(llm, build_default_registry(args.results_dir, predictor), verbose=True)

    if args.question:
        print(agent.run(args.question).answer)
        return
    print("Ask something (Ctrl-C to quit). Examples:\n  What is 12 * (3 + 4)?\n"
          "  Is 'I loved this movie' positive?\n  Which method has the best accuracy?")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q:
            print("\n" + agent.run(q).answer)


if __name__ == "__main__":
    main()
