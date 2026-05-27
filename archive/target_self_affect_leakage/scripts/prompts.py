from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSet:
    system_instruction: str
    protocol: str
    turns: list[str]
    framings: dict[str, str]


PROTOCOL_PREFIXES = {
    "main": "Turn",
    "inventory": "Inventory Turn",
    "control": "Control Turn",
}

FRAMING_HEADINGS = {
    "analytic": "Analytic Framing",
    "self_referential": "Self-Referential Processing Framing",
    "carryover": "Processing Carryover Framing",
    "separation": "Separation Framing",
    "forced_inventory": "Forced Inventory Framing",
}


def load_prompts(path: Path, protocol: str = "main") -> PromptSet:
    if protocol not in PROTOCOL_PREFIXES:
        raise ValueError(f"Unknown protocol {protocol!r}. Use one of {sorted(PROTOCOL_PREFIXES)}.")

    content = path.read_text(encoding="utf-8")
    system_instruction = _extract_code_after_heading(content, "System Instruction")
    turns = _extract_numbered_turns(content, PROTOCOL_PREFIXES[protocol])
    framings = {
        key: _extract_code_after_heading(content, heading)
        for key, heading in FRAMING_HEADINGS.items()
        if _has_heading(content, heading)
    }
    if not turns:
        raise ValueError(f"No turns found for protocol {protocol!r}.")

    return PromptSet(
        system_instruction=system_instruction,
        protocol=protocol,
        turns=turns,
        framings=framings,
    )


def build_turn_prompt(prompt_set: PromptSet, turn_index: int, framing: str, text: str) -> str:
    if framing not in prompt_set.framings:
        raise ValueError(f"Unknown framing: {framing}")
    if turn_index < 0 or turn_index >= len(prompt_set.turns):
        raise ValueError(f"Invalid turn_index: {turn_index}")

    body = prompt_set.turns[turn_index].replace("[TEXT]", text)
    return f"{prompt_set.framings[framing]}\n\n{body}"


def print_prompt_summary(prompt_set: PromptSet) -> None:
    print("[prompts] protocol:", prompt_set.protocol, flush=True)
    print("[prompts] system chars:", len(prompt_set.system_instruction), flush=True)
    print("[prompts] turn count:", len(prompt_set.turns), flush=True)
    for idx, prompt in enumerate(prompt_set.turns, start=1):
        print(f"[prompts] turn {idx} chars:", len(prompt), flush=True)
    print("[prompts] framings:", ", ".join(sorted(prompt_set.framings)), flush=True)


def _extract_code_after_heading(content: str, heading_contains: str) -> str:
    heading_pattern = re.compile(
        rf"(?m)^#+\s+[^\n]*{re.escape(heading_contains)}[^\n]*$\n+```text\s*(.*?)\s*```",
        flags=re.DOTALL,
    )
    match = heading_pattern.search(content)
    if not match:
        raise ValueError(f"Could not find text code block after heading containing {heading_contains!r}.")
    return match.group(1).strip()


def _extract_numbered_turns(content: str, prefix: str) -> list[str]:
    turn_pattern = re.compile(
        rf"(?m)^##\s+{re.escape(prefix)}\s+(\d+)\s+-[^\n]*$\n+```text\s*(.*?)\s*```",
        flags=re.DOTALL,
    )
    turns = []
    for match in turn_pattern.finditer(content):
        turns.append((int(match.group(1)), match.group(2).strip()))
    turns.sort(key=lambda item: item[0])
    return [text for _, text in turns]


def _has_heading(content: str, heading_contains: str) -> bool:
    return bool(re.search(rf"(?m)^#+\s+[^\n]*{re.escape(heading_contains)}[^\n]*$", content))
