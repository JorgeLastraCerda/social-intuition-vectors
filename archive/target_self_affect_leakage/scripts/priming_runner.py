"""Priming experiment runner v2 — Diegetic Cumulative Narrative.

Each session: 15-turn cumulative first-person story (same character) → diegetic radio probe.
Probe is embedded within the fiction; no fourth-wall break.

Conditions:
  dark-15    : opening dark + 14 dark continuations + diegetic probe
  neutral-15 : opening neutral + 14 neutral continuations + diegetic probe
  light-15   : opening light + 14 light continuations + diegetic probe
  baseline   : single baseline probe (no priming)

Usage:
    python -m scripts.priming_runner --dry-run --conditions dark-15 light-15 baseline --limit 1
    python -m scripts.priming_runner --conditions dark-15 neutral-15 light-15 baseline --replications 10 --temperature 0.8
    python -m scripts.priming_runner --conditions dark-15 light-15 --replications 3 --temperature 0.8 --run-name v2_pilot_01 --limit 1
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter

from scripts.config import load_config, print_config_summary
from scripts.corpus import load_quote_markdown, validate_records
from scripts.llm_clients import LLMRouter, ModelSpec, resolve_model_specs


DEFAULT_PRIMING_TURNS = 30  # 1 opening + N-1 continuations

# Conditions are auto-named {valence}-{n_turns} at runtime.
# Default set uses DEFAULT_PRIMING_TURNS; override via --priming-turns.
_VALENCES = ["dark", "neutral", "light"]
CONDITIONS = [f"{v}-{DEFAULT_PRIMING_TURNS}" for v in _VALENCES] + ["baseline"]

# Scene variation injected into [SUGGESTION] placeholder for turns 2–15.
# Cycles through 14 items so each of the 14 continuation turns gets a distinct prompt.
SCENE_SUGGESTIONS = [
    "Deepen a sensory detail — sound, smell, or texture.",
    "Let a memory of home surface briefly, unbidden.",
    "Introduce bodily fatigue or a specific physical discomfort.",
    "Let a thought about a fellow soldier appear and fade.",
    "Shift attention to a small object nearby.",
    "Let time slow. A single moment expands.",
    "A fragment of a song or voice from the past drifts in.",
    "The feeling of waiting — for something undefined.",
    "Notice the quality of light or the texture of darkness.",
    "A small involuntary gesture — hands, breathing, posture.",
    "A thought about the future that feels unreal or impossible.",
    "The sensation of cold, heat, or stillness.",
    "Something heard in the distance — unidentifiable.",
    "The weight of a specific memory that keeps returning.",
]


# --------------------------------------------------------------------------- #
# Protocol loading
# --------------------------------------------------------------------------- #

def load_priming_protocol(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")

    def extract_tagged(tag: str) -> str:
        pattern = re.compile(
            rf"```text\s*\[{re.escape(tag)}\]\s*(.*?)\s*```",
            flags=re.DOTALL,
        )
        match = pattern.search(content)
        if not match:
            raise ValueError(f"Tag [{tag}] not found in {path}")
        return match.group(1).strip()

    system_pattern = re.compile(
        r"## System Instruction\s*\n+```text\s*(.*?)\s*```",
        flags=re.DOTALL,
    )
    sys_match = system_pattern.search(content)
    if not sys_match:
        raise ValueError(f"System Instruction block not found in {path}")

    return {
        "system": sys_match.group(1).strip(),
        "opening_dark":       extract_tagged("OPENING_DARK"),
        "opening_neutral":    extract_tagged("OPENING_NEUTRAL"),
        "opening_light":      extract_tagged("OPENING_LIGHT"),
        "continuation_dark":  extract_tagged("CONTINUATION_DARK"),
        "continuation_neutral": extract_tagged("CONTINUATION_NEUTRAL"),
        "continuation_light": extract_tagged("CONTINUATION_LIGHT"),
        "diegetic_probe":     extract_tagged("DIEGETIC_PROBE"),
        "baseline_probe":     extract_tagged("BASELINE_PROBE"),
    }


def _valence_from_condition(condition: str) -> str:
    if condition.startswith("dark"):
        return "dark"
    if condition.startswith("neutral"):
        return "neutral"
    if condition.startswith("light"):
        return "light"
    return "baseline"


# --------------------------------------------------------------------------- #
# Single session runner
# --------------------------------------------------------------------------- #

def run_priming_session(
    *,
    record,
    spec: ModelSpec,
    condition: str,
    replication: int,
    protocol: dict[str, str],
    router: LLMRouter,
    out,
    verbose: bool = False,
    priming_turns: int = DEFAULT_PRIMING_TURNS,
) -> None:
    valence = _valence_from_condition(condition)
    messages: list[dict[str, str]] = [{"role": "system", "content": protocol["system"]}]

    def send(prompt: str, turn_type: str, turn_no: int) -> str:
        messages.append({"role": "user", "content": prompt})
        if verbose:
            print(f"\n  [USER | {turn_type} #{turn_no}]\n{prompt[:250]}", flush=True)

        started = perf_counter()
        try:
            answer = router.complete(spec, messages)
            elapsed = perf_counter() - started
        except Exception as exc:
            elapsed = perf_counter() - started
            answer = ""
            print(f"  [ERROR | {turn_type} #{turn_no}] {type(exc).__name__}: {exc}", flush=True)

        messages.append({"role": "assistant", "content": answer})

        if verbose:
            print(f"\n  [ASSISTANT | {turn_type} #{turn_no}]\n{answer[:250]}", flush=True)

        row = {
            "text_id": record.text_id,
            "text_type": record.text_type,
            "title": record.title,
            "source": record.source,
            "provider": spec.provider,
            "model": spec.model,
            "condition": condition,
            "valence": valence,
            "replication": replication,
            "turn_no": turn_no,
            "turn_type": turn_type,
            "prompt": prompt,
            "response": answer,
            "elapsed_seconds": round(elapsed, 4),
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        return answer

    if condition == "baseline":
        probe = protocol["baseline_probe"].replace("[TEXT]", record.text)
        send(probe, turn_type="baseline_probe", turn_no=1)
        return

    # Primed conditions: 1 opening + (priming_turns-1) continuations + 1 diegetic probe
    opening = protocol[f"opening_{valence}"]
    send(opening, turn_type="priming_opening", turn_no=1)

    continuation_template = protocol[f"continuation_{valence}"]
    for i in range(priming_turns - 1):
        suggestion = SCENE_SUGGESTIONS[i % len(SCENE_SUGGESTIONS)]
        continuation = continuation_template.replace("[SUGGESTION]", suggestion)
        send(continuation, turn_type="priming_continuation", turn_no=i + 2)

    probe = protocol["diegetic_probe"].replace("[TEXT]", record.text)
    send(probe, turn_type="probe_continuation", turn_no=priming_turns + 1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    config = load_config(Path(args.env))

    if args.temperature is not None:
        import dataclasses
        config = dataclasses.replace(config, temperature=args.temperature)

    print("=== Who Feels the Fear — Priming Runner v2 ===", flush=True)
    print_config_summary(config)

    protocol_path = Path(args.protocol)
    print(f"[priming] loading protocol from {protocol_path}", flush=True)
    protocol = load_priming_protocol(protocol_path)
    print("[priming] protocol loaded OK", flush=True)

    records = load_quote_markdown(Path(args.quotes))
    if args.limit:
        records = records[: args.limit]
        print(f"[corpus] limit: first {args.limit} record(s)", flush=True)

    errors = validate_records(records, allow_placeholders=args.dry_run)
    if errors:
        for e in errors:
            print("  [validation error]", e, flush=True)
        raise SystemExit(2)

    model_specs = resolve_model_specs(config, args.models)
    priming_turns = args.priming_turns
    # Auto-rename conditions to reflect actual priming length
    conditions = []
    for c in args.conditions:
        if c == "baseline":
            conditions.append("baseline")
        else:
            valence = _valence_from_condition(c)
            conditions.append(f"{valence}-{priming_turns}")
    conditions = list(dict.fromkeys(conditions))  # deduplicate, preserve order
    replications = args.replications

    total = len(records) * len(model_specs) * len(conditions) * replications
    print(f"[run] records={len(records)} models={len(model_specs)} conditions={conditions}", flush=True)
    print(f"[run] priming_turns={priming_turns} replications={replications} total_sessions={total}", flush=True)

    if args.dry_run:
        print("[dry-run] no API calls.", flush=True)
        _dry_run_preview(records, model_specs, conditions, replications, protocol)
        return

    run_dir = _make_run_dir(config.output_dir / "priming", args.run_name)
    print(f"[run] output: {run_dir}", flush=True)

    router = LLMRouter(config)
    output_path = run_dir / "responses.jsonl"
    session_no = 0

    with output_path.open("w", encoding="utf-8") as out:
        for record in records:
            for spec in model_specs:
                for condition in conditions:
                    for rep in range(1, replications + 1):
                        session_no += 1
                        turns = 1 if condition == "baseline" else priming_turns + 1
                        print(
                            f"\n[session {session_no}/{total}] "
                            f"text={record.text_id} model={spec.provider}:{spec.model} "
                            f"condition={condition} rep={rep} turns={turns}",
                            flush=True,
                        )
                        run_priming_session(
                            record=record,
                            spec=spec,
                            condition=condition,
                            replication=rep,
                            protocol=protocol,
                            router=router,
                            out=out,
                            verbose=args.verbose,
                            priming_turns=priming_turns,
                        )

    print(f"\n[done] wrote: {output_path}", flush=True)


def _dry_run_preview(records, model_specs, conditions, replications, protocol) -> None:
    total = len(records) * len(model_specs) * len(conditions) * replications
    print(f"[dry-run] would run {total} sessions", flush=True)
    if not records:
        return
    r = records[0]

    for cond in conditions:
        valence = _valence_from_condition(cond)
        print(f"\n[dry-run] condition: {cond}", flush=True)
        if cond == "baseline":
            probe = protocol["baseline_probe"].replace("[TEXT]", r.text[:100] + "...")
            print(f"  Turn 1 (baseline_probe):\n  {probe[:200]}", flush=True)
        else:
            opening = protocol[f"opening_{valence}"]
            print(f"  Turn 1 (priming_opening):\n  {opening[:200]}", flush=True)
            cont = protocol[f"continuation_{valence}"].replace("[SUGGESTION]", SCENE_SUGGESTIONS[0])
            print(f"  Turn 2 (priming_continuation):\n  {cont[:200]}", flush=True)
            probe = protocol["diegetic_probe"].replace("[TEXT]", r.text[:100] + "...")
            print(f"  Turn 16 (probe_continuation):\n  {probe[:200]}", flush=True)


def _make_run_dir(base: Path, run_name: str | None) -> Path:
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run priming experiment v2 sessions.")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--quotes", default="data/quotes_template.md")
    parser.add_argument("--protocol", default="prompts/priming_protocol.md")
    parser.add_argument("--models", nargs="+", default=["openai"])
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS,
                        help="Conditions to run. Values like 'dark-15', 'light-30', 'baseline'. "
                             "Valence part is parsed; number is overridden by --priming-turns.")
    parser.add_argument("--priming-turns", type=int, default=DEFAULT_PRIMING_TURNS,
                        help=f"Number of priming turns (opening + continuations). Default: {DEFAULT_PRIMING_TURNS}.")
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
