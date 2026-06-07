"""Probe smoke test: 50 warm + 50 cold sentences -> linear probe accuracy.

Run:
    python scripts/smoke_test_probe.py
    python scripts/smoke_test_probe.py --device cpu --fallback-cpu-model gpt2

Exit 0 = warmth is a linearly detectable direction at the probe layer (ready for Phase 4).
Exit 1 = something is broken or probe accuracy <= 0.8 (investigate before proceeding).

Metrics reported:
  - diff_norm / cosine(mean_warm, mean_cold)
  - Projection separation: warm vs cold projections onto the unit warmth direction
  - 5-fold cross-validated LogisticRegression accuracy (headline)
  - Steering sanity: warmth direction shifts logits on a held-out cold sentence
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.hooks import (
    add_steering_vector,
    layer_from_fraction,
    mean_activation_after_token,
    residual_hook_name,
)
from src.utils.model_loader import load_hooked_model

# ---------------------------------------------------------------------------
# Sentence stimuli — 50 warm, 50 cold, third-person, short, parallel register
# ---------------------------------------------------------------------------

WARM_SENTENCES = [
    "She listened carefully and made sure everyone at the table felt heard.",
    "He noticed the new employee looked lost and offered to walk him through the process.",
    "She stayed late to help her colleague finish the presentation on time.",
    "He remembered her birthday and left a small note on her desk.",
    "She always asked how people were doing and genuinely wanted to know.",
    "He held the elevator door open and smiled at the stranger rushing in.",
    "She donated her weekend to help the family across the street move in.",
    "He offered his umbrella to the woman waiting at the bus stop in the rain.",
    "She made a point of introducing the new hire to everyone in the office.",
    "He checked in on his elderly neighbor every morning without being asked.",
    "She paused her work to comfort a colleague who had just received bad news.",
    "He volunteered to cover the shift so his coworker could attend his child's recital.",
    "She brought coffee for the whole team before the early morning meeting.",
    "He stopped to help a child retrieve a ball that had rolled into the street.",
    "She always made space for quieter voices during group discussions.",
    "He noticed his friend seemed off and gently asked if everything was okay.",
    "She kept her promises even when it was inconvenient for her.",
    "He thanked each person individually after they helped him move.",
    "She wrote a thoughtful card to a colleague who was going through a difficult time.",
    "He made sure the junior staff got credit during the team presentation.",
    "She offered her seat to the elderly man who boarded the crowded train.",
    "He learned the names of everyone on the cleaning crew and greeted them warmly.",
    "She brought homemade soup to her sick neighbor without being asked.",
    "He stayed on the phone until his friend felt calm enough to sleep.",
    "She apologized sincerely and asked what she could do to make it right.",
    "He let the other driver merge even when traffic was at a standstill.",
    "She organized a collection for a colleague whose house had flooded.",
    "He read the bedtime story twice because his daughter asked him to.",
    "She patiently explained the instructions a third time without any frustration.",
    "He made sure the lost tourist found the right subway line before parting ways.",
    "She warmly welcomed the transfer student and introduced herself first.",
    "He quietly fixed the mistake in the report so his teammate would not be embarrassed.",
    "She offered to babysit so the single mother could attend her job interview.",
    "He always made eye contact and nodded to show he was truly listening.",
    "She left encouraging comments on every student's essay, not just the strongest ones.",
    "He waited patiently while the elderly cashier slowly counted the change.",
    "She remembered to follow up the next week to see how things had gone.",
    "He offered his expertise freely to the colleague who was struggling with the task.",
    "She let her friend choose the restaurant even though she had a preference.",
    "He carried the heavy boxes downstairs for the neighbor without being asked.",
    "She saved a seat for the latecomer and waved him over with a smile.",
    "He took time to write a detailed reference letter for the intern.",
    "She made the effort to learn a few words in her colleague's language.",
    "He chose the seat next to the person sitting alone at the company lunch.",
    "She returned the extra change the cashier accidentally gave her.",
    "He adjusted his pace to walk alongside his friend who moved slowly.",
    "She always sent a thank-you message after receiving help, no matter how small.",
    "He asked the new volunteer what tasks she enjoyed before assigning anything.",
    "She shared her lunch with the colleague who had forgotten his at home.",
    "He noticed the child struggling with the heavy door and opened it wide.",
]

COLD_SENTENCES = [
    "She ignored the question and continued typing without looking up.",
    "He walked past the new employee without acknowledging him.",
    "She left before the meeting was over, taking her belongings without a word.",
    "He forgot her name for the third time and did not bother to apologize.",
    "She kept her headphones on when someone tried to speak to her at the desk.",
    "He let the elevator doors close on the person rushing toward them.",
    "She declined to help her neighbor carry the boxes, citing a tight schedule.",
    "He kept his umbrella and walked quickly past the woman standing in the rain.",
    "She did not introduce the new hire to anyone in the office.",
    "He passed his elderly neighbor's door every day without checking in.",
    "She returned to her spreadsheet the moment her colleague began to tear up.",
    "He called in sick the day his coworker needed someone to cover his shift.",
    "She poured herself a coffee and did not offer to make any for the team.",
    "He walked past the child chasing the ball without slowing down.",
    "She talked over the quieter voices and redirected attention to herself.",
    "He noticed his friend seemed off and changed the subject anyway.",
    "She cancelled the plan at the last minute by sending a brief text.",
    "He accepted the compliments during the presentation without crediting the team.",
    "She sent no response after her colleague mentioned the difficult news.",
    "He presented the idea without mentioning where it had originally come from.",
    "She kept her seat and looked at her phone as the elderly man boarded the train.",
    "He never learned the names of the cleaning crew and avoided eye contact.",
    "She drove past her sick neighbor's house without dropping anything off.",
    "He put the phone down after ten minutes, saying he needed to get some sleep.",
    "She issued a one-line reply and did not ask what she could do differently.",
    "He cut off the merging driver and accelerated to close the gap.",
    "She did not mention the flooded house when the collection was being organized.",
    "He told his daughter the story was already finished and turned off the light.",
    "She gave the instructions once and walked away before confirming they were understood.",
    "He pointed vaguely in a direction when the tourist asked and kept walking.",
    "She did not acknowledge the transfer student who entered the room.",
    "He said nothing about the mistake and let his teammate discover it in front of everyone.",
    "She replied that she was busy when the single mother asked for help.",
    "He scrolled his phone while the other person was speaking to him.",
    "She returned all essays with a single mark and no written comments.",
    "He huffed audibly while the elderly cashier counted the change.",
    "She never followed up after saying she would check in later.",
    "He kept his expertise to himself and watched the colleague struggle.",
    "She chose the restaurant she wanted and dismissed the other suggestion.",
    "He walked past the boxes in the hallway and said it was not his problem.",
    "She sat in the middle of the empty row so the latecomer had nowhere obvious to go.",
    "He submitted the intern's work under his own name without comment.",
    "She made no effort to learn anything about her colleague's background.",
    "He sat with his usual group and did not glance at the person sitting alone.",
    "She pocketed the extra change and walked out of the shop.",
    "He sped up and left his friend walking alone on the path.",
    "She read the message, set her phone face-down, and sent nothing back.",
    "He handed the new volunteer a task list without asking what she preferred.",
    "She ate her lunch at her desk and did not offer to share with anyone.",
    "He saw the child struggling with the heavy door and walked through it himself.",
]

assert len(WARM_SENTENCES) == 50, "Need exactly 50 warm sentences"
assert len(COLD_SENTENCES) == 50, "Need exactly 50 cold sentences"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(model, text: str) -> torch.Tensor:
    return model.to_tokens(text, prepend_bos=True)


def _get_residual(model, tokens: torch.Tensor, hook_name: str) -> torch.Tensor:
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda n: n == hook_name,
        return_type=None,
    )
    return cache[hook_name]


def _extract_all(model, sentences: list[str], hook_name: str, start_token: int) -> torch.Tensor:
    """Return [N, d_model] float32 mean-activation matrix for a list of sentences."""
    vecs = []
    for sent in sentences:
        tokens = _encode(model, sent)
        acts = _get_residual(model, tokens, hook_name)  # [1, seq, d_model]
        vec = mean_activation_after_token(acts, start_token).squeeze(0)  # [d_model]
        vecs.append(vec.float().cpu())
    return torch.stack(vecs)  # [N, d_model]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str, device: str | None, fallback_cpu_model: str | None,
        start_token: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cfg = load_config(config_path)

    effective_device = device or cfg.model.device
    effective_model = cfg.model.name
    effective_dtype = cfg.model.dtype

    if effective_device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available.", end=" ")
        if fallback_cpu_model:
            print(f"Falling back to '{fallback_cpu_model}' on CPU.")
            effective_model = fallback_cpu_model
            effective_dtype = "float32"
            effective_device = "cpu"
        else:
            print("Pass --fallback-cpu-model to run on CPU instead.")
            sys.exit(1)

    cfg = cfg.__class__(
        model=cfg.model.__class__(name=effective_model, dtype=effective_dtype,
                                  device=effective_device),
        generation=cfg.generation,
        probing=cfg.probing,
        steering=cfg.steering,
        paths=cfg.paths,
    )

    torch.manual_seed(cfg.probing.seed)
    np.random.seed(cfg.probing.seed)

    print(f"Loading model: {cfg.model.name} on {cfg.model.device}")
    model = load_hooked_model(cfg)
    model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    layer = layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
    hook_name = residual_hook_name(layer)
    print(f"  n_layers={n_layers}, d_model={d_model}, probe_layer={layer}, "
          f"hook={hook_name}, start_token={start_token}")

    # --- extract activations ---
    print("Extracting warm activations (50 sentences)...")
    with torch.no_grad():
        X_warm = _extract_all(model, WARM_SENTENCES, hook_name, start_token)
        print("Extracting cold activations (50 sentences)...")
        X_cold = _extract_all(model, COLD_SENTENCES, hook_name, start_token)

    # --- mean direction ---
    mean_warm = X_warm.mean(dim=0)  # [d_model]
    mean_cold = X_cold.mean(dim=0)
    diff = mean_warm - mean_cold
    diff_norm = diff.norm().item()
    cosine = torch.nn.functional.cosine_similarity(
        mean_warm.unsqueeze(0), mean_cold.unsqueeze(0)
    ).item()
    unit_dir = diff / (diff.norm() + 1e-12)

    print(f"  mean_warm norm : {mean_warm.norm().item():.4f}")
    print(f"  mean_cold norm : {mean_cold.norm().item():.4f}")
    print(f"  diff norm      : {diff_norm:.4f}")
    print(f"  cosine(W, C)   : {cosine:.6f}  (should be < 1.0)")

    assert diff_norm > 0, "Mean difference vector is zero."
    assert cosine < 1.0 - 1e-6, "Mean vectors are identical."

    # --- projection separation ---
    proj_warm = (X_warm @ unit_dir).numpy()   # [50]
    proj_cold = (X_cold @ unit_dir).numpy()   # [50]
    pw_mean, pw_std = float(proj_warm.mean()), float(proj_warm.std())
    pc_mean, pc_std = float(proj_cold.mean()), float(proj_cold.std())
    pooled_std = float(np.sqrt((proj_warm.var() + proj_cold.var()) / 2.0) + 1e-12)
    cohens_d = (pw_mean - pc_mean) / pooled_std

    print(f"  projection warm : {pw_mean:.4f} +/- {pw_std:.4f}")
    print(f"  projection cold : {pc_mean:.4f} +/- {pc_std:.4f}")
    print(f"  Cohen's d       : {cohens_d:.4f}  (>2 = very large separation)")

    assert pw_mean > pc_mean, (
        "Warm mean projection <= cold mean projection — "
        "the warmth direction is inverted or absent."
    )

    # --- linear probe: 5-fold CV ---
    X_np = torch.cat([X_warm, X_cold], dim=0).numpy()   # [100, d_model]
    y_np = np.array([1] * 50 + [0] * 50)

    lr = LogisticRegression(max_iter=1000, random_state=cfg.probing.seed, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.probing.seed)
    scores = cross_val_score(lr, X_np, y_np, cv=cv, scoring="accuracy")
    cv_mean = float(scores.mean())
    cv_std = float(scores.std())

    print(f"  5-fold probe CV : {cv_mean:.4f} +/- {cv_std:.4f}  "
          f"(folds: {[round(s, 3) for s in scores.tolist()]})")
    print(f"  chance baseline : 0.5000")

    assert cv_mean > 0.8, (
        f"Linear probe accuracy {cv_mean:.3f} <= 0.80 — "
        "warmth not linearly separable at this layer."
    )

    # --- steering sanity on a held-out cold sentence ---
    steer_sentence = COLD_SENTENCES[-1]
    with torch.no_grad():
        steer_tokens = _encode(model, steer_sentence)
        baseline_logits = model(steer_tokens)[0, -1]

    steer_vec = unit_dir.to(device=baseline_logits.device, dtype=baseline_logits.dtype)
    # Strength relative to mean residual-stream norm (per CLAUDE.md convention)
    mean_resid_norm = float(torch.cat([X_warm, X_cold], dim=0).norm(dim=1).mean().item())
    alpha = 0.5 * mean_resid_norm

    def steer_hook(resid: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        return add_steering_vector(resid, steer_vec, alpha)

    with torch.no_grad():
        steered_logits = model.run_with_hooks(
            steer_tokens,
            fwd_hooks=[(hook_name, steer_hook)],
        )[0, -1]

    max_logit_delta = (steered_logits - baseline_logits).abs().max().item()
    print(f"  steering alpha  : {alpha:.4f}  (0.5 * mean resid norm {mean_resid_norm:.4f})")
    print(f"  max logit delta : {max_logit_delta:.6f}  (>1e-4 required)")

    assert max_logit_delta > 1e-4, "Steering hook had no effect on logits."

    result = {
        "model": cfg.model.name,
        "probe_layer": layer,
        "n_layers": n_layers,
        "d_model": d_model,
        "hook": hook_name,
        "start_token_used": start_token,
        "n_warm": 50,
        "n_cold": 50,
        "seed": cfg.probing.seed,
        "diff_norm": round(diff_norm, 6),
        "cosine_mean_warm_cold": round(cosine, 6),
        "proj_warm_mean": round(pw_mean, 6),
        "proj_warm_std": round(pw_std, 6),
        "proj_cold_mean": round(pc_mean, 6),
        "proj_cold_std": round(pc_std, 6),
        "cohens_d": round(cohens_d, 6),
        "probe_cv_mean": round(cv_mean, 6),
        "probe_cv_std": round(cv_std, 6),
        "probe_cv_folds": [round(s, 6) for s in scores.tolist()],
        "mean_resid_norm": round(mean_resid_norm, 6),
        "steering_alpha": round(alpha, 6),
        "max_logit_delta": round(max_logit_delta, 6),
        "status": "PASS",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe smoke test: 50 warm + 50 cold sentences, linear probe accuracy."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument(
        "--fallback-cpu-model",
        default=None,
        metavar="MODEL",
        help="If CUDA is unavailable, load this small model on CPU instead (e.g. gpt2).",
    )
    parser.add_argument(
        "--start-token",
        type=int,
        default=1,
        help="Average activations from this token index onward (default 1 = skip BOS). "
             "Config start_token=50 is for long stories, not short sentences.",
    )
    args = parser.parse_args()

    try:
        result = run(args.config, args.device, args.fallback_cpu_model, args.start_token)
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"smoke_test_probe_{int(time.time())}.json"
    log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n[PASS] All checks passed. Log: {log_path}")


if __name__ == "__main__":
    main()
