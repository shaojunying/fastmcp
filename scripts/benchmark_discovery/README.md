# Discovery benchmark: direct tools vs Tool Search vs Code Mode

This benchmark compares how a server exposes its tools to a model. It measures correctness, input tokens, model calls and latency. Discussion: https://github.com/PrefectHQ/fastmcp/issues/4902.

## What it runs

- **Tasks**: 16 tasks on a fake store backend, 4 tasks for each call depth from 1 to 4. Depth is the number of distinct tools on the longest dependent chain. Depth 4 tasks include fan-out loops. See `tasks.py`.
- **Catalog**: 10, 100 or 500 tools. The 10 base tools solve every task. Larger catalogs add 7 near-duplicates and filler tools from other domains. See `catalog.py`.
  - Some near-duplicates fail with a validation error (different parameter names).
  - Other near-duplicates take the same parameters and return stale or partial data.
- **Conditions**:

  | name | server |
  |---|---|
  | `direct` | all tools listed |
  | `oracle` | only the tools that the task needs |
  | `search_bm25` | `BM25SearchTransform()` |
  | `search_regex` | `RegexSearchTransform()` |
  | `code_mode` | `CodeMode()` |

  All transforms use their default arguments.

All tools are local and deterministic. The benchmark needs no network access.

## Metrics

- **Outcome**: `success`, `wrong_answer`, `no_answer` (the model gives up), `max_turns` or `crash` (harness failure). Accuracy excludes crashes.
- **Silent wrong tool**: a wrong answer after a backend call to a tool that the task does not need, where that call did not fail. A middleware on the server records each backend call, so calls through `call_tool` and `execute` count too.
- **Input tokens**: the sum over every model call, including discovery calls. Tool definitions count again on each call. The harness counts with `o200k_base`. When a model adapter returns provider usage, the report uses the provider numbers.
- **Model calls, tool calls, discovery calls**, and latency (wall clock for real models, simulated for the scripted policy).

## Statistics

Every condition runs the same tasks. `stats.py` pairs each condition with `direct` and with `oracle` by task:

- the difference in accuracy and in silent-wrong-tool rate, with a 95% bootstrap interval that resamples tasks,
- the number of tasks that got better or worse, with an exact sign test,
- the geometric mean of the per-task token ratio, with a bootstrap interval,
- pass^k with the unbiased estimate C(c, k) / C(n, k).

## Scripted policy

`ScriptedModel` replaces the model for offline runs. It is not a model of LLM behaviour:

- It knows the reference plan of each task.
- It picks tools perfectly when it can see them.
- When it must search, it takes the top hit. After an error, it tries the next hit (3 candidates at most).

So scripted results show only what the server exposes and how search ranks results. They say nothing about model accuracy. `--noise P` takes the second hit with probability P, to exercise the pass^k and variance reporting.

## Run

Use the commands below from the repository root. The harness needs `tiktoken`, which is not a FastMCP dependency.

```bash
uv run --with tiktoken python scripts/benchmark_discovery/run.py --repeats 3 --k 3 --out results/scripted
uv run --with tiktoken python scripts/benchmark_discovery/run.py --repeats 1 --k 1 --sizes 10,20,30,40,50,70,100 --out results/crossover
uv run --with tiktoken --with pytest pytest -p no:cacheprovider -c /dev/null --rootdir scripts/benchmark_discovery scripts/benchmark_discovery/tests
```

Each run writes `episodes.jsonl` (one line per episode), `summary.json` and `report.md`. The full scripted grid takes about 40 seconds.

## Real models

A real model adapter implements the `Model` protocol in `harness.py`:

- `start(task, condition)` resets state.
- `complete(system, messages, tools)` returns a `Completion` with text or tool calls, and provider token usage when available.

No adapter is included yet. Real-model runs cost money and do not belong in CI.

## Scripted results (main at 5ac27a2)

Input tokens of each condition divided by `direct`, mean over the 4 tasks of each depth:

| depth | condition | 10 | 20 | 30 | 40 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|---|
| 1 | `search_bm25` | 1.01 | 0.60 | 0.42 | 0.33 | 0.27 | 0.19 | 0.14 |
| 4 | `search_bm25` | 2.70 | 1.71 | 1.26 | 1.01 | 0.84 | 0.63 | 0.45 |
| 1 | `code_mode` | 1.68 | 1.02 | 0.74 | 0.59 | 0.50 | 0.39 | 0.31 |
| 4 | `code_mode` | 1.01 | 0.80 | 0.65 | 0.56 | 0.51 | 0.45 | 0.41 |

The catalog size where discovery starts to save tokens changes with task depth. Search results stay in the history, so deeper tasks resend more of them.
