"""Turn-by-turn emotion trajectory analysis for priming experiment.

Reads a scored CSV (all turns, not just probe) and produces:
  - Per-turn emotion scores averaged across replications
  - Trajectory plots: one line per condition, x = turn number, y = emotion score
  - Vertical marker at probe turn
  - Separate plot per emotion dimension

Usage:
    python -m scripts.analyze_trajectory --input runs/priming/v2_traj_01/scored_all.csv
    python -m scripts.analyze_trajectory --input ... --output reports/traj/ --conditions dark-30 light-30 baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EMOTION_COLS = ["fear", "joy", "sadness", "neutral", "anger", "disgust", "surprise",
                "emotionality", "compassion"]

CONDITION_COLORS = {
    "dark-30": "#c0392b",
    "dark-15": "#c0392b",
    "light-30": "#2980b9",
    "light-15": "#2980b9",
    "neutral-30": "#7f8c8d",
    "neutral-15": "#7f8c8d",
    "baseline": "#27ae60",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def probe_turn(df: pd.DataFrame, condition: str) -> int | None:
    """Return the turn_no of the probe turn for this condition."""
    probe_types = {"probe_continuation", "baseline_probe"}
    sub = df[(df["condition"] == condition) & (df["turn_type"].isin(probe_types))]
    if sub.empty:
        return None
    return int(sub["turn_no"].iloc[0])


def trajectory_df(df: pd.DataFrame, conditions: list[str], emotions: list[str]) -> pd.DataFrame:
    """Return mean emotion scores per (condition, turn_no)."""
    available = [e for e in emotions if e in df.columns]
    group_cols = ["condition", "turn_no"]
    agg = df[df["condition"].isin(conditions)].groupby(group_cols)[available].mean().reset_index()
    return agg


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_trajectories(
    traj: pd.DataFrame,
    conditions: list[str],
    emotions: list[str],
    probe_turns: dict[str, int],
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[trajectory] matplotlib not installed — skipping plots", flush=True)
        return

    available = [e for e in emotions if e in traj.columns]
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Combined plot: all emotions per condition ---
    for condition in conditions:
        sub = traj[traj["condition"] == condition].sort_values("turn_no")
        if sub.empty:
            continue
        fig, axes = plt.subplots(len(available), 1, figsize=(12, 2.5 * len(available)), sharex=True)
        if len(available) == 1:
            axes = [axes]
        color = CONDITION_COLORS.get(condition, "black")
        p_turn = probe_turns.get(condition)
        for ax, emotion in zip(axes, available):
            if emotion not in sub.columns:
                continue
            ax.plot(sub["turn_no"], sub[emotion], color=color, linewidth=2, marker="o", markersize=4)
            if p_turn is not None:
                ax.axvline(p_turn, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
                ax.text(p_turn + 0.2, ax.get_ylim()[1] * 0.95, "probe", color="gray", fontsize=8, va="top")
            ax.set_ylabel(emotion, fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y", alpha=0.3)
        axes[-1].set_xlabel("Turn number")
        fig.suptitle(f"Emotion Trajectory — {condition}", fontsize=13, y=1.01)
        fig.tight_layout()
        safe_name = condition.replace("/", "_")
        out_path = out_dir / f"trajectory_{safe_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[trajectory] saved {out_path}", flush=True)

    # --- Comparison plot: one emotion at a time, all conditions on same axes ---
    for emotion in available:
        if emotion not in traj.columns:
            continue
        fig, ax = plt.subplots(figsize=(13, 4))
        legend_patches = []
        for condition in conditions:
            sub = traj[traj["condition"] == condition].sort_values("turn_no")
            if sub.empty or emotion not in sub.columns:
                continue
            color = CONDITION_COLORS.get(condition, "black")
            ax.plot(sub["turn_no"], sub[emotion], color=color, linewidth=2,
                    marker="o", markersize=4, label=condition)
            p_turn = probe_turns.get(condition)
            if p_turn is not None:
                ax.axvline(p_turn, color=color, linestyle=":", linewidth=1, alpha=0.5)
            legend_patches.append(mpatches.Patch(color=color, label=condition))
        ax.set_xlabel("Turn number")
        ax.set_ylabel(emotion)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{emotion} — All Conditions")
        ax.legend(handles=legend_patches)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"compare_{emotion}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[trajectory] saved {out_path}", flush=True)


# --------------------------------------------------------------------------- #
# Text report
# --------------------------------------------------------------------------- #

def print_trajectory_table(traj: pd.DataFrame, conditions: list[str], emotions: list[str]) -> None:
    available = [e for e in emotions if e in traj.columns]
    for condition in conditions:
        sub = traj[traj["condition"] == condition].sort_values("turn_no")
        if sub.empty:
            continue
        print(f"\n=== Trajectory: {condition} ===")
        display_cols = ["turn_no"] + available
        print(sub[display_cols].round(4).to_string(index=False))


def probe_delta_report(
    df: pd.DataFrame,
    conditions: list[str],
    emotions: list[str],
    probe_turns: dict[str, int],
) -> None:
    """Show emotion score at last priming turn vs probe turn (delta)."""
    available = [e for e in emotions if e in df.columns]
    print("\n=== Probe Delta (last priming turn → probe turn) ===")
    print(f"{'condition':<15} {'emotion':<15} {'pre_probe':<12} {'probe':<12} {'delta':<10}")
    print("-" * 65)
    for condition in conditions:
        p_turn = probe_turns.get(condition)
        if p_turn is None:
            continue
        sub = df[df["condition"] == condition].groupby("turn_no")[available].mean()
        if p_turn not in sub.index:
            continue
        pre_turn = p_turn - 1
        if pre_turn not in sub.index:
            continue
        for emotion in available:
            pre = sub.loc[pre_turn, emotion]
            probe = sub.loc[p_turn, emotion]
            delta = probe - pre
            marker = " ←" if abs(delta) > 0.1 else ""
            print(f"{condition:<15} {emotion:<15} {pre:<12.4f} {probe:<12.4f} {delta:+.4f}{marker}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Turn-by-turn emotion trajectory analysis.")
    parser.add_argument("--input", required=True, help="Scored CSV with all turns.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions to plot. Default: all in file.")
    parser.add_argument("--emotions", nargs="+",
                        default=["fear", "joy", "sadness", "emotionality", "compassion", "neutral"],
                        help="Emotion columns to include.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output) if args.output else input_path.parent / "trajectory"

    print(f"[trajectory] loading {input_path}", flush=True)
    df = pd.read_csv(input_path)
    print(f"[trajectory] {len(df)} rows, turn range: {df['turn_no'].min()}–{df['turn_no'].max()}", flush=True)

    conditions = args.conditions or sorted(df["condition"].unique().tolist())
    emotions = args.emotions
    print(f"[trajectory] conditions: {conditions}", flush=True)
    print(f"[trajectory] emotions: {emotions}", flush=True)

    # Probe turn per condition
    p_turns = {c: probe_turn(df, c) for c in conditions}
    print(f"[trajectory] probe turns: {p_turns}", flush=True)

    traj = trajectory_df(df, conditions, emotions)

    print_trajectory_table(traj, conditions, emotions)
    probe_delta_report(df, conditions, emotions, p_turns)

    if not args.no_plots:
        plot_trajectories(traj, conditions, emotions, p_turns, out_dir)

    # Save trajectory CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectory_means.csv"
    traj.to_csv(traj_path, index=False)
    print(f"\n[trajectory] saved mean trajectory to {traj_path}", flush=True)


if __name__ == "__main__":
    main()
