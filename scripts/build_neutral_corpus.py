#!/usr/bin/env python3
"""
build_neutral_corpus.py
-----------------------
Build a socially neutral text corpus (Wikipedia article introductions) for the
PCA valence-denoising step. This matches the Anthropic emotion-concepts paper,
which removed generic variance directions using a neutral corpus.

Why Wikipedia intros: factual, externally sourced (not written by an LLM, so no
circularity with the model we probe), and free of social evaluation. We length
match them to the concept stories so length cannot leak into the PCA.

Network: streams the Hugging Face `wikimedia/wikipedia` dataset, so run it where
there is internet (the SCCKN login node), NOT on a GPU compute node.

    pip install datasets
    PYTHONPATH=. python scripts/build_neutral_corpus.py --config config/config.yaml

Offline check of the filtering logic (no network):
    PYTHONPATH=. python scripts/build_neutral_corpus.py --self-test

Output: data/stimuli/neutral_corpus.jsonl  ->  {id, text, title, source, n_words}
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from src.utils.config import load_config

# Skip intros carrying social / affective valence so the corpus stays neutral
# (METHOD_NOTES: avoid crime, opinion, and other valence-laden material).
VALENCE_STOP = re.compile(
    r"\b(war|wars|battle|killed|kill|murder|massacre|genocide|death|died|disaster|"
    r"attack|terror|abuse|slaver|rape|victim|crime|criminal|tragedy|atrocit|famine|"
    r"epidemic|pandemic|crisis|scandal|controvers|riot|hero|heroic|beloved|brilliant|"
    r"notorious|infamous|tyrann|brutal)\b",
    re.I,
)


def n_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def make_intro(text: str, max_words: int) -> str:
    """Take leading paragraphs up to ~max_words, then trim to a sentence boundary."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    acc, wc = [], 0
    for p in paras:
        acc.append(p)
        wc += n_words(p)
        if wc >= max_words:
            break
    intro = " ".join(acc)
    if n_words(intro) > max_words:
        sents = re.split(r"(?<=[.!?])\s+", intro)
        out, w = [], 0
        for s in sents:
            if out and w + n_words(s) > max_words:
                break
            out.append(s)
            w += n_words(s)
        intro = " ".join(out)
    return intro.strip()


def is_candidate(title: str, intro: str, lo: int, hi: int) -> bool:
    if not intro or "(disambiguation)" in title.lower():
        return False
    if title.lower().startswith(("list of", "index of", "timeline of")):
        return False
    w = n_words(intro)
    if w < lo or w > hi:
        return False
    if intro.count(".") < 1:           # require prose, not a stub
        return False
    if VALENCE_STOP.search(intro):
        return False
    return True


def build(cfg) -> None:
    from datasets import load_dataset

    nf = cfg.neutral
    lo, hi, target = nf.min_words, nf.max_words, nf.n_texts
    pool_target = target * 4           # gather a surplus, then seed-sample

    print(f"[stream] {nf.source_dataset}:{nf.source_config}  target={target} "
          f"(pool {pool_target}), words {lo}-{hi}")
    ds = load_dataset(nf.source_dataset, nf.source_config, split="train", streaming=True)

    cands, seen = [], set()
    for rec in ds:
        title = rec.get("title", "")
        if title in seen:
            continue
        intro = make_intro(rec.get("text", ""), hi)
        if is_candidate(title, intro, lo, hi):
            seen.add(title)
            cands.append((title, intro))
            if len(cands) % 500 == 0:
                print(f"  [collect] {len(cands)}/{pool_target}")
        if len(cands) >= pool_target:
            break

    rng = random.Random(cfg.probing.seed)
    rng.shuffle(cands)
    chosen = cands[:target]

    out_path = Path(cfg.paths.stimuli) / "neutral_corpus.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, (title, intro) in enumerate(chosen):
            f.write(json.dumps({
                "id": f"neutral_{i:05d}", "text": intro, "title": title,
                "source": f"{nf.source_dataset}:{nf.source_config}",
                "n_words": n_words(intro),
            }, ensure_ascii=False) + "\n")

    ws = [n_words(t[1]) for t in chosen]
    print(f"[done] wrote {len(chosen)} neutral texts -> {out_path}")
    if ws:
        print(f"       words min/mean/max = {min(ws)}/{round(sum(ws)/len(ws))}/{max(ws)}")


def self_test() -> None:
    samples = [
        ("Photosynthesis", "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. " * 6),
        ("World War II", "World War II was a global war that killed millions and was a tragedy of immense scale. " * 6),
        ("List of rivers", "This article lists rivers. " * 6),
        ("Granite", "Granite is a coarse grained igneous rock composed mostly of quartz and feldspar. It forms from slowly cooling magma. " * 6),
        ("Stub", "Short."),
    ]
    expect = {"Photosynthesis": True, "World War II": False, "List of rivers": False,
              "Granite": True, "Stub": False}
    ok = True
    for title, text in samples:
        intro = make_intro(text, 200)
        got = is_candidate(title, intro, 90, 200)
        flag = "OK" if got == expect[title] else "FAIL"
        if got != expect[title]:
            ok = False
        print(f"  [{flag}] {title}: candidate={got} (expected {expect[title]}), words={n_words(intro)}")
    print("SELF-TEST", "PASS" if ok else "FAIL")


def parse_args():
    ap = argparse.ArgumentParser(description="Build neutral Wikipedia corpus for denoising.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--self-test", action="store_true", help="offline logic check, no network")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    build(load_config(args.config))


if __name__ == "__main__":
    main()
