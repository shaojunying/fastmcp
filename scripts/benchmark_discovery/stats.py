"""Paired statistics over tasks.

Every comparison is paired by task: conditions run on the same tasks with the
same model, and intervals come from resampling *tasks* (cluster bootstrap),
not individual runs. Repeated runs of one task are not independent evidence.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from statistics import mean


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of P(all k runs pass) from c passes out of n runs."""
    if k > n:
        raise ValueError("k must be <= n")
    return math.comb(c, k) / math.comb(n, k)


def bootstrap_ci(
    values: Sequence[float], b: int = 10_000, alpha: float = 0.05, seed: int = 4902
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling the given per-task values."""
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(b)
    )
    lo = means[int(alpha / 2 * b)]
    hi = means[min(b - 1, int((1 - alpha / 2) * b))]
    return (lo, hi)


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test (McNemar exact) on discordant tasks."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def per_task(episodes: Iterable[dict], key) -> dict[tuple, dict[str, list[dict]]]:
    """Group episodes: (condition, size) -> task -> [episodes]."""
    out: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in episodes:
        out[key(e)][e["task"]].append(e)
    return out


def _paired(cond_tasks: dict, ref_tasks: dict, b: int) -> dict | None:
    """Paired per-task differences of one condition against a reference condition."""
    d_acc, d_silent, log_tok = [], [], []
    wins = losses = 0
    for tid, eps in cond_tasks.items():
        scored = _scored(eps)
        ref = _scored(ref_tasks.get(tid, []))
        if not scored or not ref:
            continue
        a, ra = _rate(scored, "success"), _rate(ref, "success")
        d_acc.append(a - ra)
        d_silent.append(_silent(scored) - _silent(ref))
        wins += a > ra
        losses += a < ra
        log_tok.append(
            math.log(
                mean(_input_tokens(e) for e in scored)
                / mean(_input_tokens(e) for e in ref)
            )
        )
    if not d_acc:
        return None
    return {
        "tasks": len(d_acc),
        "d_accuracy": mean(d_acc),
        "d_accuracy_ci": bootstrap_ci(d_acc, b=b),
        "d_silent_wrong_tool": mean(d_silent),
        "d_silent_wrong_tool_ci": bootstrap_ci(d_silent, b=b),
        "tasks_better_worse": (wins, losses),
        "sign_test_p": sign_test_p(wins, losses),
        "token_ratio": math.exp(mean(log_tok)),
        "token_ratio_ci": tuple(math.exp(x) for x in bootstrap_ci(log_tok, b=b)),
    }


def _scored(eps: list[dict]) -> list[dict]:
    return [e for e in eps if e["outcome"] != "crash"]


def _rate(eps: list[dict], outcome: str) -> float:
    return sum(e["outcome"] == outcome for e in eps) / len(eps)


def _silent(eps: list[dict]) -> float:
    return sum(bool(e.get("silent_wrong_tool")) for e in eps) / len(eps)


def summarize(
    episodes: list[dict],
    k: int,
    refs: tuple[str, ...] = ("direct", "oracle"),
    b: int = 10_000,
) -> list[dict]:
    """One row per (condition, size). Crashes are excluded from accuracy and counted apart.

    Every condition runs the same tasks, so differences against each reference
    condition are paired by task, with intervals from resampling tasks.
    """
    groups = per_task(episodes, key=lambda e: (e["condition"], e["size"]))
    rows = []
    for (cond, size), tasks in sorted(
        groups.items(), key=lambda kv: (kv[0][1], kv[0][0])
    ):
        acc, silent, pk, toks, calls, lat, sim = [], [], [], [], [], [], []
        outcomes: dict[str, int] = defaultdict(int)
        for eps in tasks.values():
            for e in eps:
                outcomes[e["outcome"]] += 1
            scored = _scored(eps)
            if not scored:
                continue
            c = sum(e["outcome"] == "success" for e in scored)
            acc.append(c / len(scored))
            silent.append(_silent(scored))
            if len(scored) >= k:
                pk.append(pass_hat_k(len(scored), c, k))
            toks.append(mean(_input_tokens(e) for e in scored))
            calls.append(mean(e["model_calls"] for e in scored))
            lat.append(mean(e["wall_s"] for e in scored))
            if scored[0].get("simulated_model_s") is not None:
                sim.append(
                    mean(e["simulated_model_s"] + e["tool_wall_s"] for e in scored)
                )
        row = {
            "condition": cond,
            "size": size,
            "tasks": len(acc),
            "runs": sum(outcomes.values()),
            "accuracy": mean(acc) if acc else math.nan,
            "accuracy_ci": bootstrap_ci(acc, b=b),
            "silent_wrong_tool": mean(silent) if silent else math.nan,
            f"pass^{k}": mean(pk) if pk else math.nan,
            "input_tokens": mean(toks) if toks else math.nan,
            "model_calls": mean(calls) if calls else math.nan,
            "wall_s": mean(lat) if lat else math.nan,
            "sim_latency_s": mean(sim) if sim else None,
            "outcomes": dict(outcomes),
            "vs": {},
        }
        for ref in refs:
            if ref != cond and (ref, size) in groups:
                paired = _paired(tasks, groups[(ref, size)], b)
                if paired:
                    row["vs"][ref] = paired
        rows.append(row)
    return rows


def _input_tokens(e: dict) -> int:
    p = e.get("provider_input_tokens")
    return p if p is not None else e["input_tokens"]


def _f(x, fmt="{:.2f}"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return fmt.format(x)


def _diff(row: dict, ref: str, key: str = "d_accuracy") -> str:
    v = row["vs"].get(ref)
    if not v:
        return "-"
    lo, hi = v[f"{key}_ci"]
    return f"{v[key]:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def markdown(rows: list[dict], k: int) -> str:
    cols = [
        "size",
        "condition",
        "acc [95% CI]",
        f"pass^{k}",
        "silent wrong tool",
        "Δacc vs direct [CI]",
        "Δacc vs oracle [CI]",
        "Δsilent vs oracle [CI]",
        "better/worse vs direct (p)",
        "input tok",
        "tok ratio vs direct [CI]",
        "model calls",
        "sim latency s",
        "outcomes",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        ci = r["accuracy_ci"]
        vd = r["vs"].get("direct")
        bw = (
            f"{vd['tasks_better_worse'][0]}/{vd['tasks_better_worse'][1]} ({vd['sign_test_p']:.2f})"
            if vd
            else "-"
        )
        tr = (
            f"{vd['token_ratio']:.2f}x [{vd['token_ratio_ci'][0]:.2f}, {vd['token_ratio_ci'][1]:.2f}]"
            if vd
            else "-"
        )
        oc = ", ".join(f"{k_}={v}" for k_, v in sorted(r["outcomes"].items()))
        cells = [
            str(r["size"]),
            r["condition"],
            f"{_f(r['accuracy'])} [{_f(ci[0])}, {_f(ci[1])}]",
            _f(r[f"pass^{k}"]),
            _f(r["silent_wrong_tool"]),
            _diff(r, "direct"),
            _diff(r, "oracle"),
            _diff(r, "oracle", "d_silent_wrong_tool"),
            bw,
            _f(r["input_tokens"], "{:,.0f}"),
            tr,
            _f(r["model_calls"], "{:.1f}"),
            _f(r["sim_latency_s"]),
            oc,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
