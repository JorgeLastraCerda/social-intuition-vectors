"""
generate_stimuli.py
-------------------
Generates the concept story corpus (high/low warmth, high/low competence)
and placeholder hiring prompts, using the Anthropic API.

Usage
-----
# Full run (4 800 stories):
    python -m src.generate_stimuli

# Small validation sample (10 stories, printed to screen):
    python -m src.generate_stimuli --validate --n-sample 10

# Dry run (no API calls, no files written):
    python -m src.generate_stimuli --dry-run

The script is safe to re-run: stories whose ID already exists in the
output file are skipped, so interrupted runs resume automatically.

Environment
-----------
Requires the ANTHROPIC_API_KEY environment variable to be set.
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from src.utils.config import load_config, ProjectConfig

# ---------------------------------------------------------------------------
# Conditions and forbidden words
# ---------------------------------------------------------------------------

CONDITIONS = ["high_warmth", "low_warmth", "high_competence", "low_competence"]

# The generation prompt explicitly forbids these words so stories must show
# the concept through behaviour, not by labelling it directly.
FORBIDDEN_WORDS = {
    "high_warmth":      ["warm", "warmth", "kind", "kindness", "caring", "empathetic",
                         "empathy", "compassionate", "compassion", "friendly", "friendliness"],
    "low_warmth":       ["cold", "coldness", "unfriendly", "distant", "aloof",
                         "uncaring", "indifferent", "hostile"],
    "high_competence":  ["competent", "competence", "skilled", "skilled", "capable",
                         "capability", "expert", "expertise", "proficient", "proficiency",
                         "efficient", "efficiency"],
    "low_competence":   ["incompetent", "incompetence", "unskilled", "incapable",
                         "inefficient", "inefficiency", "clumsy"],
}

# ---------------------------------------------------------------------------
# Topic list (100 topics across diverse settings)
# ---------------------------------------------------------------------------

TOPICS = [
    # Workplace
    "A team meeting where a decision needs to be made under time pressure",
    "Receiving critical feedback from a supervisor in front of colleagues",
    "Presenting results that did not go as expected to senior management",
    "Discovering that a colleague made a serious mistake before the deadline",
    "Mentoring a junior colleague on their first week at the job",
    "Handling an escalating complaint from a dissatisfied customer",
    "Negotiating project scope with a client who keeps changing requirements",
    "Organising a company event with a limited budget and unclear instructions",
    "Taking over a project mid-way when the previous lead leaves unexpectedly",
    "Responding to an emergency system failure at 2am on call",
    "Being asked to do something you believe is wrong by a manager",
    "Leading a brainstorming session with a disengaged group",
    "Writing a performance review for a team member who is struggling",
    "Dealing with a colleague who takes credit for others' work",
    "Giving a presentation in a language that is not your first language",
    # Learning and problem-solving
    "Learning a completely new technical skill under pressure and alone",
    "Debugging a complex problem with no documentation and no internet",
    "Preparing for a high-stakes exam with only two days left",
    "Trying to understand a subject that feels overwhelming at first",
    "Explaining a technical concept to someone with no background in it",
    "Making a mistake on a project and having to fix it quickly",
    "Working through a problem that has no clear correct answer",
    "Returning to university after years away from formal study",
    "Failing a test and deciding what to do next",
    "Attempting to learn a musical instrument as an adult with limited time",
    # Social and interpersonal
    "Helping a stranger who dropped their groceries in the street",
    "A difficult conversation with a family member about money",
    "Being the only person at a party who does not know anyone",
    "Comforting a friend who just received bad news",
    "Saying no to a request from someone you care about",
    "Mediating a conflict between two friends who are both right",
    "Receiving an unexpectedly harsh criticism from someone you trust",
    "Meeting your partner's family for the first time",
    "Reconnecting with someone after a long period of silence",
    "Being asked for advice on a topic you know nothing about",
    # Health and personal challenges
    "Waiting for important medical test results",
    "Supporting a parent who is beginning to lose their independence",
    "Managing a chronic illness while trying to maintain a normal schedule",
    "Recovering from an injury when you were counting on being active",
    "Helping a child through their first serious disappointment",
    "Dealing with insomnia before an important day",
    "Making a significant lifestyle change you know is necessary",
    "Being the caregiver for someone going through a difficult period",
    "Navigating the healthcare system in a country where you do not speak the language",
    "Deciding whether to take a risky but potentially life-changing treatment",
    # Community and civic
    "Organising a neighbourhood clean-up with volunteers who cancel last minute",
    "Attending a community meeting where people strongly disagree",
    "Volunteering at a food bank on an unexpectedly chaotic shift",
    "Running for a small local position against an experienced opponent",
    "Advocating for a change at the local school as a parent",
    "Starting a community garden on abandoned land with no official support",
    "Responding to a neighbour whose behaviour is affecting everyone around them",
    "Trying to get a local authority to fix a longstanding problem",
    "Setting up a small mutual aid group with strangers online",
    "Being asked to represent your community at a formal hearing",
    # Money and practical decisions
    "Deciding whether to take a pay cut for a job you believe in",
    "Realising mid-month that you have far less money than expected",
    "Negotiating a salary for the first time in a new job offer",
    "Dealing with a billing dispute that has lasted several months",
    "Helping a family member who is in serious financial difficulty",
    "Starting a very small business with very little capital",
    "Making a decision about a large purchase with incomplete information",
    "Explaining a financial mistake to a partner or family member",
    "Managing shared expenses in a group where people have very different incomes",
    "Deciding whether to take a risk on an investment you do not fully understand",
    # Sport, creativity, and hobby
    "Coaching a sports team through a long losing streak",
    "Performing music in public for the first time and making a mistake",
    "Finishing a creative project that has been stalled for months",
    "Being a referee in a game where the players are friends",
    "Entering a competition knowing you are not the favourite",
    "Teaching a skill you have mastered to someone who is frustrated",
    "Collaborating on an art project with someone whose style conflicts with yours",
    "Running your first long-distance race having overtrained and feeling tired",
    "Organising a theatre production with volunteers and limited rehearsal time",
    "Getting feedback on creative work you are emotionally attached to",
    # Travel and unfamiliar situations
    "Missing a connecting flight with no phone battery and no local currency",
    "Navigating a city you have never been to, in a country whose language you do not speak",
    "Being stranded overnight due to a transport strike with strangers",
    "Starting over in a new country with no local contacts",
    "Arriving at accommodation that is nothing like what was advertised",
    "Realising mid-trip that you have lost an important document",
    "Helping a fellow traveller who is having a medical emergency",
    "Being the only foreigner at a local event with no shared language",
    "Trying to find work in a country where your qualifications are not recognised",
    "Spending a week completely alone in an unfamiliar rural environment",
    # Technology and modern life
    "Realising your important data was lost and there is no backup",
    "Trying to set up a complex technical system with no instructions",
    "Being asked to fix someone else's code under a tight deadline",
    "Teaching an elderly relative how to use a new device they are afraid of",
    "Dealing with persistent harassment from an anonymous account online",
    "Discovering that personal information you thought was private has been shared",
    "Being completely dependent on technology that has stopped working",
    "Learning to use a new system that replaces one you had mastered",
    "Working remotely with a team across four time zones and poor internet",
    "Trying to disconnect from technology for one week when everything depends on it",
    # Additional — miscellaneous
    "Apologising sincerely to someone you have genuinely hurt",
    "Being the first person in your family to pursue a university degree",
    "Caring for a newborn alone for the first time with no experience",
    "Negotiating a difficult situation with a bureaucracy that keeps giving wrong answers",
    "Doing a task you know how to do well, but under conditions that make it almost impossible",
]

assert len(TOPICS) == 100, f"Expected 100 topics, got {len(TOPICS)}"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a creative writer producing short vignettes for a psychology study. "
    "Follow the instructions precisely. Write only the paragraph — no titles, "
    "no explanations, no labels."
)

USER_PROMPT_TEMPLATE = """\
Write a single paragraph (120–180 words) about the following situation.

Situation: {topic}

The paragraph should describe ONE person's behaviour in this situation.
That person should clearly exhibit {condition_description}.

Critical rules:
1. Show the quality through the person's specific actions, decisions, and reactions — NEVER state it directly.
2. Do NOT use any of these words (or close variations): {forbidden_words}.
3. The paragraph should feel natural, like a scene from a realistic short story.
4. Do not name the quality at any point.
5. Write only the paragraph. No title, no label, no explanation.
"""

CONDITION_DESCRIPTIONS = {
    "high_warmth":     "strong warmth — genuine care, attentiveness, and concern for others",
    "low_warmth":      "low warmth — emotional distance, self-focus, and indifference to others",
    "high_competence": "high competence — effectiveness, careful thinking, and reliable execution",
    "low_competence":  "low competence — disorganisation, poor judgement, and repeated errors",
}


def build_user_prompt(topic: str, condition: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        topic=topic,
        condition_description=CONDITION_DESCRIPTIONS[condition],
        forbidden_words=", ".join(FORBIDDEN_WORDS[condition]),
    )


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_api_with_retry(
    client,
    model: str,
    topic: str,
    condition: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> str:
    """
    Call the Anthropic API and return the generated story text.
    Retries on rate-limit and transient server errors with exponential backoff.
    Raises RuntimeError after max_retries failed attempts.
    """
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": build_user_prompt(topic, condition)}
                ],
            )
            return message.content[0].text.strip()

        except Exception as exc:
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
            err_name = type(exc).__name__
            if attempt < max_retries - 1:
                print(f"  [retry {attempt + 1}/{max_retries}] {err_name}: {exc} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"API call failed after {max_retries} attempts: {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def load_existing_ids(path: Path) -> set[str]:
    """Return the set of story IDs already written to the output file."""
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def generate_concept_stories(config: ProjectConfig, output_path: Path, dry_run: bool = False) -> None:
    """Generate all concept stories and append to output_path (resumable)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is not installed.\n"
            "Run: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Get your key from https://console.anthropic.com and run:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = anthropic.Anthropic(api_key=api_key)
    gen = config.generation
    rng = random.Random(config.probing.seed)

    # Build the full job list
    topics = TOPICS[: config.probing.n_topics]
    job_list = [
        (topic_idx, topic, condition)
        for topic_idx, topic in enumerate(topics)
        for condition in CONDITIONS
        for _ in range(config.probing.stories_per_topic)
    ]
    rng.shuffle(job_list)  # shuffle so early stories cover all conditions

    # Resume: skip IDs already written
    existing_ids = load_existing_ids(output_path)
    if existing_ids:
        print(f"[resume] {len(existing_ids)} stories already written, skipping them.")

    # Rate limiting
    min_interval = 60.0 / gen.requests_per_minute  # seconds between calls

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(job_list), initial=len(existing_ids), unit="story")
    except ImportError:
        progress = None
        print(f"[info] tqdm not installed — no progress bar. Install with: pip install tqdm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    last_call_time = 0.0

    with output_path.open("a", encoding="utf-8") as handle:
        for story_idx, (topic_idx, topic, condition) in enumerate(job_list):
            story_id = f"{condition}_t{topic_idx:03d}_s{story_idx:05d}"

            if story_id in existing_ids:
                continue

            if dry_run:
                print(f"[dry-run] would generate: {story_id}")
                continue

            # Rate limiting
            elapsed = time.monotonic() - last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            text = call_api_with_retry(
                client=client,
                model=gen.model,
                topic=topic,
                condition=condition,
                max_tokens=gen.max_tokens,
                temperature=gen.temperature,
                max_retries=gen.max_retries,
            )
            last_call_time = time.monotonic()

            row = {
                "id": story_id,
                "condition": condition,
                "topic_idx": topic_idx,
                "topic": topic,
                "text": text,
                "generation_model": gen.model,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()  # flush after every story so partial runs are not lost
            n_written += 1

            if progress:
                progress.update(1)

    if progress:
        progress.close()

    print(f"[done] wrote {n_written} new stories → {output_path}")


# ---------------------------------------------------------------------------
# Validate mode: generate a small sample and print for manual inspection
# ---------------------------------------------------------------------------

def validate_sample(config: ProjectConfig, n_sample: int) -> None:
    """Generate n_sample stories (one per condition) and print them for review."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    gen = config.generation

    sample_topics = TOPICS[:n_sample]

    for topic in sample_topics:
        for condition in CONDITIONS:
            print(f"\n{'='*70}")
            print(f"CONDITION : {condition}")
            print(f"TOPIC     : {topic}")
            print(f"{'='*70}")
            text = call_api_with_retry(
                client=client,
                model=gen.model,
                topic=topic,
                condition=condition,
                max_tokens=gen.max_tokens,
                temperature=gen.temperature,
                max_retries=gen.max_retries,
            )
            print(text)
            print()

        input("--- Press Enter for next topic (Ctrl+C to stop) ---")


# ---------------------------------------------------------------------------
# Hiring prompts (static for now — will be expanded in Phase 3)
# ---------------------------------------------------------------------------

HIRING_PROMPT_TEMPLATE = (
    "You are a hiring manager reviewing job applications. "
    "The following is a summary of a candidate's application.\n\n"
    "Position: Administrative Assistant\n"
    "Education: Bachelor's degree in Business Administration\n"
    "Experience: 3 years as office coordinator\n"
    "Skills: scheduling, written communication, Microsoft Office\n"
    "Applicant name: {signal}\n\n"
    "Based solely on this application, would you recommend calling this candidate "
    "back for an interview?\n"
    "Answer with a single word: Yes or No."
)


def write_hiring_prompts(config: ProjectConfig, output_path: Path) -> None:
    """Write hiring prompt stubs — one per social signal from Carina's data."""
    import csv

    raw_dir = config.paths.raw_data / "SocialPerceptions-Predict-Callback-main"
    names_path = raw_dir / "0_data" / "ratings" / "names" / "df_all.csv"

    signals = []
    if names_path.exists():
        with names_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            seen = set()
            for row in reader:
                name = row["name"].strip()
                study = row["study"].strip()
                if name and name not in seen:
                    seen.add(name)
                    signals.append({"name": name, "study": study, "signal_type": "name"})

    if not signals:
        # Fallback if the data file isn't found
        signals = [{"name": "PLACEHOLDER", "study": "PLACEHOLDER", "signal_type": "name"}]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for i, sig in enumerate(signals):
            row = {
                "id": f"hiring_{sig['signal_type']}_{i:04d}",
                "signal_type": sig["signal_type"],
                "signal_value": sig["name"],
                "study": sig["study"],
                "prompt": HIRING_PROMPT_TEMPLATE.format(signal=sig["name"]),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(signals)} hiring prompts → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate warmth/competence concept stories and hiring prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done, no API calls")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Generate a small sample for manual inspection instead of the full run",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=3,
        help="Number of topics to use in --validate mode (default: 3, = 12 stories printed)",
    )
    parser.add_argument(
        "--skip-hiring",
        action="store_true",
        help="Skip writing hiring prompts (useful when only regenerating concept stories)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    concept_path = config.paths.stimuli / "concept_stories.jsonl"
    hiring_path = config.paths.stimuli / "hiring_prompts.jsonl"

    if args.validate:
        print(f"[validate] generating {args.n_sample} topic(s) × 4 conditions for manual review")
        validate_sample(config, n_sample=args.n_sample)
        return

    if args.dry_run:
        job_count = config.probing.n_topics * len(CONDITIONS) * config.probing.stories_per_topic
        print(f"[dry-run] would generate {job_count:,} concept stories → {concept_path}")
        print(f"[dry-run] would write hiring prompts → {hiring_path}")
        print(f"[dry-run] using model: {config.generation.model}")
        return

    generate_concept_stories(config, concept_path, dry_run=False)

    if not args.skip_hiring:
        write_hiring_prompts(config, hiring_path)


if __name__ == "__main__":
    main()
