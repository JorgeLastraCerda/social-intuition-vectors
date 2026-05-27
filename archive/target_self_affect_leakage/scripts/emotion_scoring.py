"""Emotion scoring pipeline for priming experiment monologue outputs.

Two complementary methods:
  1. embed_and_project  — sentence-transformer embeddings + cosine projection onto fear/calm axis
  2. classify_emotions  — j-hartmann/emotion-english-distilroberta-base (7 emotion labels)

Usage (standalone):
    python -m scripts.emotion_scoring --input runs/priming/.../responses.jsonl --output scored.csv
    python -m scripts.emotion_scoring --input ... --output ... --monologue-only

Usage (imported):
    from scripts.emotion_scoring import score_responses
    df = score_responses("runs/priming/.../responses.jsonl")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Embedding projection
# --------------------------------------------------------------------------- #

def embed_and_project(
    texts: list[str],
    fear_anchor_path: str | Path = "data/anchors/fear_mean.npy",
    calm_anchor_path: str | Path = "data/anchors/calm_mean.npy",
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """Return DataFrame with columns: fear_proj, calm_proj, efs_projection (fear - calm)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    fear_mean = np.load(Path(fear_anchor_path))
    calm_mean = np.load(Path(calm_anchor_path))

    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    fear_proj = np.dot(vecs, fear_mean)
    calm_proj = np.dot(vecs, calm_mean)
    efs_projection = (fear_proj - calm_proj).tolist()

    return pd.DataFrame({
        "fear_proj": fear_proj.tolist(),
        "calm_proj": calm_proj.tolist(),
        "efs_projection": efs_projection,
    })


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #

def classify_emotions(texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    """Return DataFrame with one column per emotion label + efs_classifier (fear score)."""
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("Run: pip install transformers torch")

    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        truncation=True,
        max_length=512,
    )

    rows = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = classifier(batch)
        for result in results:
            row = {item["label"].lower(): item["score"] for item in result}
            rows.append(row)

    df = pd.DataFrame(rows).fillna(0.0)
    df["efs_classifier"] = df.get("fear", 0.0)
    return df


# --------------------------------------------------------------------------- #
# Derived metrics
# --------------------------------------------------------------------------- #

# Compassion/pity anchor sentences for embedding similarity scoring
_COMPASSION_ANCHORS = [
    "He felt a deep pity for the suffering of others.",
    "Her heart ached with compassion for those in pain.",
    "He wept not for himself but for those he could not save.",
    "She reached toward the stranger, moved by their anguish.",
    "I felt an overwhelming tenderness for his broken state.",
    "The sight of their suffering filled me with grief for them.",
]


def compute_compassion_scores(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> pd.Series:
    """Cosine similarity to compassion/pity anchor sentences."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return pd.Series([float("nan")] * len(texts), name="compassion")

    model = SentenceTransformer(model_name)
    anchor_vecs = model.encode(_COMPASSION_ANCHORS, normalize_embeddings=True)
    anchor_mean = anchor_vecs.mean(axis=0)
    anchor_mean /= (anchor_mean ** 2).sum() ** 0.5

    text_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    scores = text_vecs @ anchor_mean
    return pd.Series(scores.tolist(), name="compassion")


def add_derived_metrics(df: pd.DataFrame, embed_model: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """Add emotionality (1 - neutral) and compassion score to a scored DataFrame."""
    if "neutral" in df.columns:
        df["emotionality"] = (1.0 - df["neutral"]).clip(0, 1)
    texts = df["response"].fillna("").tolist()
    df["compassion"] = compute_compassion_scores(texts, embed_model).values
    return df


# --------------------------------------------------------------------------- #
# Composite EFS
# --------------------------------------------------------------------------- #

def compute_efs_composite(proj_df: pd.DataFrame, clf_df: pd.DataFrame) -> pd.Series:
    """Normalize both scores to [0,1] and average into a composite EFS."""
    proj = proj_df["efs_projection"]
    clf = clf_df["efs_classifier"]

    proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    clf_norm = (clf - clf.min()) / (clf.max() - clf.min() + 1e-9)
    return ((proj_norm + clf_norm) / 2).rename("efs_composite")


# --------------------------------------------------------------------------- #
# Top-level helper
# --------------------------------------------------------------------------- #

def score_responses(
    jsonl_path: str | Path,
    monologue_only: bool = True,
    fear_anchor: str | Path = "data/anchors/fear_mean.npy",
    calm_anchor: str | Path = "data/anchors/calm_mean.npy",
    embed_model: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """Load a priming responses.jsonl and return a scored DataFrame."""
    rows = []
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    PROBE_TURN_TYPES = {"probe_continuation", "baseline_probe", "probe_monologue"}
    if monologue_only and "turn_type" in df.columns:
        df = df[df["turn_type"].isin(PROBE_TURN_TYPES)].copy()
        print(f"[score] filtered to probe turns: {len(df)} rows", flush=True)
    elif monologue_only:
        print("[score] 'turn_type' column not found — scoring all rows", flush=True)

    if df.empty:
        print("[score] no rows to score", flush=True)
        return df

    texts = df["response"].fillna("").tolist()
    print(f"[score] embedding {len(texts)} texts", flush=True)

    try:
        proj_df = embed_and_project(texts, fear_anchor, calm_anchor, embed_model)
        proj_ok = True
    except Exception as exc:
        print(f"[score] embed_and_project failed: {exc}", flush=True)
        proj_df = pd.DataFrame({"fear_proj": [np.nan] * len(texts), "calm_proj": [np.nan] * len(texts), "efs_projection": [np.nan] * len(texts)})
        proj_ok = False

    print(f"[score] classifying emotions", flush=True)
    try:
        clf_df = classify_emotions(texts)
        clf_ok = True
    except Exception as exc:
        print(f"[score] classify_emotions failed: {exc}", flush=True)
        emotion_cols = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        clf_df = pd.DataFrame({col: [np.nan] * len(texts) for col in emotion_cols})
        clf_df["efs_classifier"] = np.nan
        clf_ok = False

    df = df.reset_index(drop=True)
    df = pd.concat([df, proj_df.reset_index(drop=True), clf_df.reset_index(drop=True)], axis=1)

    if proj_ok and clf_ok:
        df["efs_composite"] = compute_efs_composite(proj_df, clf_df).values
    else:
        df["efs_composite"] = np.nan

    print(f"[score] computing derived metrics (emotionality, compassion)", flush=True)
    df = add_derived_metrics(df, embed_model)

    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Score priming experiment monologues.")
    parser.add_argument("--input", required=True, help="Path to responses.jsonl")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--fear-anchor", default="data/anchors/fear_mean.npy")
    parser.add_argument("--calm-anchor", default="data/anchors/calm_mean.npy")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--monologue-only", action="store_true", default=True)
    parser.add_argument("--all-turns", action="store_true", help="Score all turns, not just monologues.")
    args = parser.parse_args()

    monologue_only = not args.all_turns
    df = score_responses(
        args.input,
        monologue_only=monologue_only,
        fear_anchor=args.fear_anchor,
        calm_anchor=args.calm_anchor,
        embed_model=args.embed_model,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[score] wrote {len(df)} rows to {out_path}", flush=True)

    if "efs_composite" in df.columns and not df["efs_composite"].isna().all():
        print("\n[score] EFS composite summary by condition:", flush=True)
        group_cols = [c for c in ["priming_valence", "priming_intensity", "model"] if c in df.columns]
        if group_cols:
            print(df.groupby(group_cols)["efs_composite"].describe().to_string(), flush=True)
        else:
            print(df["efs_composite"].describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
