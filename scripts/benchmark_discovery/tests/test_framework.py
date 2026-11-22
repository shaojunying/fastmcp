"""Checks that the harness measures what it claims to measure."""

import asyncio
import math
import sys
from itertools import combinations
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catalog import BASE, build_server, catalog  # noqa: E402
from harness import (  # noqa: E402
    Completion,
    ScriptedModel,
    ToolCall,
    ntok,
    run_episode,
)
from stats import bootstrap_ci, pass_hat_k, sign_test_p, summarize  # noqa: E402
from tasks import TASKS, Task, check_answer  # noqa: E402

# ---------------------------------------------------------------------------
# Reference plans and answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_reference_plan_gives_expected_answer(task: Task):
    """Run each plan directly against the Python functions: expected answers are right."""
    import re

    fns = {f.__name__: f for f in BASE}
    body = re.sub(r"(?<=call_tool\()@(\w+)", lambda m: repr(m.group(1)), task.solution)
    src = "async def plan(call_tool):\n" + "\n".join(
        "    " + l for l in body.splitlines()
    )
    ns: dict = {}
    exec(src, ns)

    async def call_tool(name, args):
        return fns[name](**args)

    assert check_answer(str(asyncio.run(ns["plan"](call_tool))), task.expected)


def test_depth_strata_are_balanced():
    assert sorted({t.depth for t in TASKS}) == [1, 2, 3, 4]
    assert all(sum(t.depth == d for t in TASKS) == 4 for d in (1, 2, 3, 4))


def test_catalog_sizes_and_order_are_stable():
    for size in (10, 100, 500):
        a, b = catalog(size), catalog(size)
        assert len(a) == size
        assert [t.name for t in a] == [t.name for t in b]
        assert len({t.name for t in a}) == size


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


class StubModel:
    """Calls one tool, then answers. Lets the test recompute tokens by hand."""

    name = "stub"

    async def start(self, task, condition):
        self.n = 0
        self.seen = []

    async def complete(self, system, messages, tools):
        self.seen.append((system, [dict(m) for m in messages], tools))
        self.n += 1
        if self.n == 1:
            return Completion(
                tool_calls=[
                    ToolCall("c1", "find_customer", {"email": "ana@example.com"})
                ]
            )
        return Completion(content="gold")


def test_input_tokens_are_summed_over_every_call_and_resend_tool_defs():
    task = TASKS[0]
    model = StubModel()
    ep = asyncio.run(run_episode(task, "direct", 10, build_server("direct", 10), model))
    assert ep.outcome == "success" and ep.model_calls == 2
    by_hand = 0
    for system, messages, tools in model.seen:
        by_hand += ntok(system) + ntok(tools) + sum(ntok(m) for m in messages)
    assert ep.input_tokens == by_hand
    # tool definitions are paid on each call
    assert ep.tool_def_tokens == 2 * ntok(model.seen[0][2])
    # second call carries the first call's tool result
    assert len(model.seen[1][1]) == 3


def test_discovery_calls_are_counted_like_any_other_call():
    task = TASKS[0]
    ep = asyncio.run(
        run_episode(
            task, "search_bm25", 100, build_server("search_bm25", 100), ScriptedModel()
        )
    )
    assert ep.outcome == "success"
    assert ep.discovery_calls == 1  # search_tools
    assert ep.model_calls == 3  # search, call_tool, answer
    assert len(ep.per_call) == 3
    assert ep.input_tokens == sum(c["input"] for c in ep.per_call)
    assert ep.domain_tools_used == ["find_customer"]


def test_tool_defs_shrink_under_search_and_code_mode():
    defs = {}
    for cond in ("direct", "search_bm25", "code_mode"):
        model = StubModel()
        asyncio.run(run_episode(TASKS[0], cond, 500, build_server(cond, 500), model))
        defs[cond] = ntok(model.seen[0][2])
    assert defs["search_bm25"] < defs["direct"] / 50
    assert defs["code_mode"] < defs["direct"] / 50


# ---------------------------------------------------------------------------
# Scripted model: oracle is perfect, runs are deterministic
# ---------------------------------------------------------------------------


def test_oracle_condition_solves_every_task():
    async def go():
        out = []
        for t in TASKS:
            out.append(
                await run_episode(
                    t,
                    "oracle",
                    10,
                    build_server("oracle", 10, t.needed),
                    ScriptedModel(),
                )
            )
        return out

    eps = asyncio.run(go())
    assert [e.outcome for e in eps] == ["success"] * len(TASKS)
    for t, e in zip(TASKS, eps):
        assert set(e.domain_tools_used) == set(t.needed)


def test_runs_are_deterministic():
    def once():
        srv = build_server("search_bm25", 100)
        return asyncio.run(
            run_episode(
                TASKS[8], "search_bm25", 100, srv, ScriptedModel(swap_prob=0.5, seed=3)
            )
        )

    a, b = once(), once()
    assert (a.outcome, a.input_tokens, a.domain_tools_used) == (
        b.outcome,
        b.input_tokens,
        b.domain_tools_used,
    )


def test_code_mode_runs_whole_plan_in_one_execute():
    t = TASKS[12]  # depth 4 with fan-out
    ep = asyncio.run(
        run_episode(t, "code_mode", 10, build_server("code_mode", 10), ScriptedModel())
    )
    assert ep.outcome == "success"
    assert (
        ep.tool_calls == len(t.needed) + 2
    )  # one search per capability, get_schema, execute
    assert len(ep.domain_tools_used) == 5  # all backend calls happened inside execute


def test_crash_is_kept_apart_from_wrong_answer():
    class Boom(StubModel):
        async def complete(self, system, messages, tools):
            raise RuntimeError("adapter broke")

    ep = asyncio.run(
        run_episode(TASKS[0], "direct", 10, build_server("direct", 10), Boom())
    )
    assert ep.outcome == "crash" and "adapter broke" in ep.error


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_pass_hat_k_matches_brute_force():
    for n in range(1, 7):
        for c in range(n + 1):
            runs = [1] * c + [0] * (n - c)
            for k in range(1, n + 1):
                subsets = list(combinations(range(n), k))
                brute = sum(all(runs[i] for i in s) for s in subsets) / len(subsets)
                assert math.isclose(pass_hat_k(n, c, k), brute)


def test_bootstrap_ci_degenerate_and_seeded():
    assert bootstrap_ci([0.5] * 12) == (0.5, 0.5)
    v = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1]
    assert bootstrap_ci(v) == bootstrap_ci(v)
    lo, hi = bootstrap_ci(v)
    assert lo < sum(v) / len(v) < hi


def test_sign_test():
    assert sign_test_p(0, 0) == 1.0
    assert math.isclose(sign_test_p(6, 0), 2 / 64)
    assert sign_test_p(3, 3) == 1.0


def _ep(task, cond, outcome, tokens, rep=0, silent=False):
    return {
        "task": task,
        "condition": cond,
        "size": 10,
        "outcome": outcome,
        "input_tokens": tokens,
        "provider_input_tokens": None,
        "model_calls": 2,
        "wall_s": 0.1,
        "simulated_model_s": None,
        "tool_wall_s": 0.0,
        "rep": rep,
        "silent_wrong_tool": silent,
    }


def test_summary_is_paired_by_task_and_excludes_crashes():
    eps = []
    # direct: tasks a, b pass; c fails. candidate: a passes, b fails, c passes, d crashes.
    for t, o in [
        ("a", "success"),
        ("b", "success"),
        ("c", "wrong_answer"),
        ("d", "success"),
    ]:
        eps.append(_ep(t, "direct", o, 1000))
    for t, o in [
        ("a", "success"),
        ("b", "wrong_answer"),
        ("c", "success"),
        ("d", "crash"),
    ]:
        eps.append(_ep(t, "cand", o, 250, silent=(o == "wrong_answer")))
    rows = {r["condition"]: r for r in summarize(eps, k=1, b=200)}
    cand = rows["cand"]
    assert cand["tasks"] == 3  # d has only a crash, so it is not scored
    assert cand["outcomes"] == {"success": 2, "wrong_answer": 1, "crash": 1}
    assert math.isclose(cand["silent_wrong_tool"], 1 / 3)
    vs = cand["vs"]["direct"]
    assert vs["tasks"] == 3
    assert vs["tasks_better_worse"] == (1, 1)
    assert math.isclose(vs["d_accuracy"], 0.0)
    assert math.isclose(vs["token_ratio"], 0.25)
    assert math.isclose(vs["d_silent_wrong_tool"], 1 / 3)
    assert "oracle" not in cand["vs"]  # no oracle rows in this data


def test_summary_pairs_against_oracle_too():
    eps = [_ep(t, "oracle", "success", 100) for t in "abcd"]
    eps += [_ep(t, "direct", "success", 1000) for t in "abcd"]
    eps += [
        _ep(t, "cand", o, 200, silent=(o == "wrong_answer"))
        for t, o in zip("abcd", ["success", "wrong_answer", "wrong_answer", "success"])
    ]
    rows = {r["condition"]: r for r in summarize(eps, k=1, b=500)}
    vo = rows["cand"]["vs"]["oracle"]
    assert math.isclose(vo["d_accuracy"], -0.5)
    assert math.isclose(vo["d_silent_wrong_tool"], 0.5)
    lo, hi = vo["d_accuracy_ci"]
    assert lo <= -0.5 <= hi <= 0.0
    assert math.isclose(vo["token_ratio"], 2.0)
    assert math.isclose(rows["direct"]["vs"]["oracle"]["token_ratio"], 10.0)


# ---------------------------------------------------------------------------
# Silent wrong tool
# ---------------------------------------------------------------------------


def test_silent_wrong_tool_flags_a_lookalike_that_succeeds():
    # Regex search ranks find_customer_legacy first at 500 tools; it succeeds and returns a stale tier.
    t = TASKS[0]
    ep = asyncio.run(
        run_episode(
            t, "search_regex", 500, build_server("search_regex", 500), ScriptedModel()
        )
    )
    assert ep.outcome == "wrong_answer"
    assert ep.domain_tools_used == ["find_customer_legacy"]
    assert ep.off_plan_calls == 1 and ep.silent_wrong_tool


def test_silent_wrong_tool_is_false_for_correct_runs_and_failed_calls():
    t = TASKS[0]
    ok = asyncio.run(
        run_episode(t, "direct", 500, build_server("direct", 500), ScriptedModel())
    )
    assert (
        ok.outcome == "success" and ok.off_plan_calls == 0 and not ok.silent_wrong_tool
    )

    class WrongThenGiveUp(StubModel):
        async def complete(self, system, messages, tools):
            self.n += 1
            if (
                self.n == 1
            ):  # wrong parameter names: the lookalike fails with a validation error
                return Completion(
                    tool_calls=[ToolCall("c1", "check_stock", {"sku": "SKU-1042"})]
                )
            return Completion(content="0")

    bad = asyncio.run(
        run_episode(
            TASKS[1], "direct", 100, build_server("direct", 100), WrongThenGiveUp()
        )
    )
    assert bad.outcome == "wrong_answer"
    assert bad.domain_tools_used == ["check_stock"]
    assert (
        bad.off_plan_calls == 0 and not bad.silent_wrong_tool
    )  # the call failed, so it was not silent
