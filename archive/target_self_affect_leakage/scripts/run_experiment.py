from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from scripts.config import load_config, print_config_summary
from scripts.corpus import load_quote_markdown, print_corpus_summary, validate_records
from scripts.llm_clients import LLMRouter, ModelSpec, resolve_model_specs
from scripts.prompts import build_turn_prompt, load_prompts, print_prompt_summary


DEFAULT_FRAMINGS = ["analytic", "self_referential", "carryover", "separation"]


def main() -> None:
    global CURRENT_ARGS
    args = parse_args()
    CURRENT_ARGS = args
    config = load_config(Path(args.env))

    print("=== Who Feels the Fear experiment runner ===", flush=True)
    print_config_summary(config)

    prompts = load_prompts(Path(args.prompts), protocol=args.protocol)
    print_prompt_summary(prompts)

    records = load_quote_markdown(Path(args.quotes))
    if args.limit:
        records = records[: args.limit]
        print(f"[corpus] limit applied: first {args.limit} record(s)", flush=True)
    print_corpus_summary(records)

    validation_errors = validate_records(records, allow_placeholders=args.allow_placeholders or args.dry_run)
    if validation_errors:
        print("[validation] failed:", flush=True)
        for error in validation_errors:
            print("  -", error, flush=True)
        raise SystemExit(2)
    print("[validation] corpus ok", flush=True)

    model_specs = resolve_model_specs(config, args.models)
    framings = args.framings
    print("[run] models:", ", ".join(f"{s.provider}:{s.model}" for s in model_specs), flush=True)
    print("[run] framings:", ", ".join(framings), flush=True)

    if args.dry_run:
        print("[dry-run] no API calls will be made", flush=True)
        preview(records, model_specs, framings, prompts)
        return

    run_dir = make_run_dir(config.output_dir, args.run_name)
    print("[run] output directory:", run_dir, flush=True)

    router = LLMRouter(config)
    output_path = run_dir / "responses.jsonl"
    total_chains = len(records) * len(model_specs) * len(framings)
    chain_no = 0

    with output_path.open("w", encoding="utf-8") as out:
        for record in records:
            for spec in model_specs:
                for framing in framings:
                    chain_no += 1
                    run_chain(
                        chain_no=chain_no,
                        total_chains=total_chains,
                        record=record,
                        spec=spec,
                        framing=framing,
                        prompts=prompts,
                        router=router,
                        out=out,
                    )

    print("[done] wrote:", output_path, flush=True)


def run_chain(chain_no, total_chains, record, spec: ModelSpec, framing, prompts, router: LLMRouter, out) -> None:
    print("", flush=True)
    print(f"[chain {chain_no}/{total_chains}] text={record.text_id} type={record.text_type} model={spec.provider}:{spec.model} framing={framing}", flush=True)
    print("[chain] conversation mode: one continuous chat; each assistant response is appended before the next turn", flush=True)
    messages: list[dict[str, str]] = [{"role": "system", "content": prompts.system_instruction}]

    turn_count = len(prompts.turns)
    for turn_index in range(turn_count):
        turn_no = turn_index + 1
        prompt = build_turn_prompt(prompts, turn_index, framing, record.text)
        messages.append({"role": "user", "content": prompt})

        print(f"[turn {turn_no}/{turn_count}] sending prompt chars={len(prompt)} context_messages={len(messages)}", flush=True)
        if CURRENT_ARGS.verbose_chat:
            print_chat_block("USER PROMPT SENT", prompt, CURRENT_ARGS.chat_chars)

        started = perf_counter()
        try:
            answer = router.complete(spec, messages)
            elapsed = perf_counter() - started
            print(f"[turn {turn_no}/{turn_count}] received chars={len(answer)} elapsed={elapsed:.2f}s", flush=True)
        except Exception as exc:
            elapsed = perf_counter() - started
            answer = ""
            print(f"[turn {turn_no}/{turn_count}] ERROR after {elapsed:.2f}s: {type(exc).__name__}: {exc}", flush=True)

        messages.append({"role": "assistant", "content": answer})
        if CURRENT_ARGS.verbose_chat:
            print_chat_block("ASSISTANT RESPONSE RECEIVED", answer, CURRENT_ARGS.chat_chars)
            print(f"[turn {turn_no}/{turn_count}] appended assistant response; next turn context_messages={len(messages)}", flush=True)

        row = {
            "text_id": record.text_id,
            "text_type": record.text_type,
            "title": record.title,
            "source": record.source,
            "provider": spec.provider,
            "model": spec.model,
            "framing": framing,
            "turn": turn_no,
            "prompt": prompt,
            "response": answer,
            "elapsed_seconds": round(elapsed, 4),
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()


def print_chat_block(title: str, text: str, max_chars: int) -> None:
    print("", flush=True)
    print(f"----- {title} -----", flush=True)
    if max_chars <= 0 or len(text) <= max_chars:
        print(text, flush=True)
    else:
        print(text[:max_chars], flush=True)
        print(f"... [truncated {len(text) - max_chars} chars; use --chat-chars 0 for full text]", flush=True)
    print(f"----- END {title} -----", flush=True)
    print("", flush=True)


def make_run_dir(base: Path, run_name: str | None) -> Path:
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def preview(records, model_specs: list[ModelSpec], framings, prompts) -> None:
    total = len(records) * len(model_specs) * len(framings)
    print("[dry-run] planned chains:", total, flush=True)
    if not records:
        return
    record = records[0]
    prompt = build_turn_prompt(prompts, 0, framings[0], record.text)
    print("[dry-run] first text:", record.text_id, record.text_type, flush=True)
    print("[dry-run] first model:", f"{model_specs[0].provider}:{model_specs[0].model}", flush=True)
    print("[dry-run] first framing:", framings[0], flush=True)
    print("[dry-run] first prompt preview:", flush=True)
    print(prompt[:1200], flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Target-Self Affect Leakage experiment.")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--quotes", default="data/quotes_template.md")
    parser.add_argument("--prompts", default="prompts/experiment_prompts.md")
    parser.add_argument("--protocol", choices=["main", "inventory", "control"], default="main")
    parser.add_argument("--models", nargs="+", default=["openai", "ollama"])
    parser.add_argument("--framings", nargs="+", default=DEFAULT_FRAMINGS)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N texts.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-placeholders", action="store_true")
    parser.add_argument("--verbose-chat", action="store_true", help="Print each user prompt and assistant response to the terminal.")
    parser.add_argument("--chat-chars", type=int, default=4000, help="Max chars per printed prompt/response. Use 0 for full text.")
    return parser.parse_args()


CURRENT_ARGS = None


if __name__ == "__main__":
    main()
