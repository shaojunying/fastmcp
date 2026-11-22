"""Run the discovery benchmark.

    uv run --with tiktoken python scripts/benchmark_discovery/run.py --repeats 3 --out results/scripted
    uv run --with tiktoken python scripts/benchmark_discovery/run.py --noise 0.2 --repeats 5 --out results/noisy

Writes ``episodes.jsonl`` (one line per episode) and ``report.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from catalog import CONDITIONS, SIZES, build_server
from harness import ScriptedModel, episode_dict, run_episode
from stats import markdown, summarize
from tasks import TASKS


def make_model(args):
    if args.model == "scripted":
        return ScriptedModel(swap_prob=args.noise, seed=args.seed)
    raise SystemExit(
        f"unknown model {args.model!r}; real-model adapters are not wired yet"
    )


async def main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = make_model(args)
    conditions = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else list(SIZES)
    tasks = [t for t in TASKS if not args.tasks or t.id in args.tasks.split(",")]
    episodes = []
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for size in sizes:
            for cond in conditions:
                shared = None if cond == "oracle" else build_server(cond, size)
                for task in tasks:
                    server = shared or build_server(cond, size, task.needed)
                    for rep in range(args.repeats):
                        model.seed = args.seed + rep
                        ep = await run_episode(task, cond, size, server, model, rep=rep)
                        d = episode_dict(ep)
                        episodes.append(d)
                        f.write(json.dumps(d) + "\n")
                print(f"size={size:<4} {cond:<13} done", flush=True)
    rows = summarize(episodes, k=args.k)
    (out / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = markdown(rows, k=args.k)
    (out / "report.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    logging.getLogger("fastmcp").setLevel(logging.ERROR)
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="scripted")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", default="")
    p.add_argument("--sizes", default="")
    p.add_argument("--tasks", default="")
    p.add_argument("--out", default="results/scripted")
    asyncio.run(main(p.parse_args()))
