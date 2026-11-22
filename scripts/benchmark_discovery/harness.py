"""Episode runner, token accounting and the scripted model.

One episode = one task under one condition with one model. The runner owns
the agent loop, so every model call (including discovery calls such as
``search_tools`` / ``get_schema``) is counted the same way in every condition.

Token accounting (``count_call_tokens``) is the harness's own estimate:
o200k_base over the canonical JSON of the tool definitions, the system
prompt and every message, as resent on each call. A real model adapter also
records the provider-reported usage next to it; reports use the provider
numbers when present.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import tiktoken
from tasks import SEARCH_QUERY, SEARCH_REGEX, TOOL_REF, Task, check_answer

from fastmcp import Client, FastMCP

SYSTEM_PROMPT = (
    "You are a store operations assistant. Use the available tools to answer. "
    "When you have the answer, reply with the answer only."
)

_ENC = tiktoken.get_encoding("o200k_base")


def ntok(obj: Any) -> int:
    text = (
        obj
        if isinstance(obj, str)
        else json.dumps(obj, separators=(",", ":"), sort_keys=True)
    )
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Completion:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    simulated_latency_s: float | None = None  # scripted model only


class Model(Protocol):
    name: str

    async def start(self, task: Task, condition: str) -> None: ...

    async def complete(
        self, system: str, messages: list[dict], tools: list[dict]
    ) -> Completion: ...


def tool_defs(mcp_tools) -> list[dict]:
    """OpenAI-style function definitions, exactly as a client would send them."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema,
            },
        }
        for t in mcp_tools
    ]


def count_call_tokens(
    system: str, messages: list[dict], tools: list[dict]
) -> dict[str, int]:
    t = {
        "system": ntok(system),
        "tool_defs": ntok(tools) if tools else 0,
        "messages": sum(ntok(m) for m in messages),
    }
    t["input"] = t["system"] + t["tool_defs"] + t["messages"]
    return t


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

OUTCOMES = ("success", "wrong_answer", "no_answer", "max_turns", "crash")


@dataclass
class Episode:
    task: str
    depth: int
    condition: str
    size: int
    model: str
    rep: int
    outcome: str = "crash"
    answer: str | None = None
    model_calls: int = 0
    tool_calls: int = 0  # every tools/call, including discovery and execute
    domain_tools_used: list[str] = field(
        default_factory=list
    )  # backend tools actually run
    off_plan_calls: int = 0  # successful backend calls to a tool the task does not need
    silent_wrong_tool: bool = (
        False  # wrong answer after an off-plan call that did not fail
    )
    discovery_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    tool_def_tokens: int = 0
    output_tokens: int = 0
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    wall_s: float = 0.0
    tool_wall_s: float = 0.0
    simulated_model_s: float | None = None
    per_call: list[dict] = field(default_factory=list)
    error: str | None = None


DISCOVERY = {"search_tools", "search", "get_schema", "tags", "list_tools"}
PROXY = {"call_tool", "execute"}


def result_text(res) -> str:
    parts = [c.text for c in res.content if getattr(c, "type", None) == "text"]
    return "\n".join(parts)


async def run_episode(
    task: Task,
    condition: str,
    size: int,
    server: FastMCP,
    model: Model,
    rep: int = 0,
    max_turns: int = 30,
) -> Episode:
    ep = Episode(task.id, task.depth, condition, size, model.name, rep)
    ran: list[str] = []
    ran_ok: list[bool] = []
    t0 = time.perf_counter()
    try:
        from fastmcp.server.middleware import Middleware

        class Spy(Middleware):
            async def on_call_tool(self, context, call_next):
                name = context.message.name
                if name in DISCOVERY or name in PROXY:
                    return await call_next(context)
                ran.append(name)
                ran_ok.append(False)
                result = await call_next(context)  # raises on tool errors
                ran_ok[-1] = not getattr(result, "is_error", False)
                return result

        spy = Spy()
        server.add_middleware(spy)
        try:
            await model.start(task, condition)
            async with Client(server) as client:
                tools = tool_defs(await client.list_tools())
                messages: list[dict] = [{"role": "user", "content": task.prompt}]
                for _ in range(max_turns):
                    tok = count_call_tokens(SYSTEM_PROMPT, messages, tools)
                    comp = await model.complete(SYSTEM_PROMPT, messages, tools)
                    out_msg = _assistant_message(comp)
                    out_tok = ntok({k: v for k, v in out_msg.items() if k != "role"})
                    ep.model_calls += 1
                    ep.input_tokens += tok["input"]
                    ep.tool_def_tokens += tok["tool_defs"]
                    ep.output_tokens += out_tok
                    if comp.provider_input_tokens is not None:
                        ep.provider_input_tokens = (
                            ep.provider_input_tokens or 0
                        ) + comp.provider_input_tokens
                    if comp.provider_output_tokens is not None:
                        ep.provider_output_tokens = (
                            ep.provider_output_tokens or 0
                        ) + comp.provider_output_tokens
                    if comp.simulated_latency_s is not None:
                        ep.simulated_model_s = (
                            ep.simulated_model_s or 0.0
                        ) + comp.simulated_latency_s
                    ep.per_call.append(
                        {**tok, "output": out_tok, "n_tool_calls": len(comp.tool_calls)}
                    )
                    messages.append(out_msg)
                    if not comp.tool_calls:
                        ep.answer = comp.content
                        break
                    for tc in comp.tool_calls:
                        ep.tool_calls += 1
                        if tc.name in DISCOVERY:
                            ep.discovery_calls += 1
                        ts = time.perf_counter()
                        res = await client.call_tool_mcp(tc.name, tc.arguments)
                        ep.tool_wall_s += time.perf_counter() - ts
                        if res.is_error:
                            ep.tool_errors += 1
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_text(res),
                            }
                        )
                else:
                    ep.outcome = "max_turns"
        finally:
            server.middleware.remove(spy)
        if ep.outcome != "max_turns":
            if ep.answer is None or ep.answer.startswith("UNABLE"):
                ep.outcome = "no_answer"
            elif check_answer(ep.answer, task.expected):
                ep.outcome = "success"
            else:
                ep.outcome = "wrong_answer"
    except Exception as exc:  # noqa: BLE001  # harness failure, kept apart from wrong answers
        ep.outcome = "crash"
        ep.error = "".join(traceback.format_exception_only(exc)).strip()[:500]
    ep.domain_tools_used = ran
    needed = set(task.needed)
    ep.off_plan_calls = sum(ok and name not in needed for name, ok in zip(ran, ran_ok))
    ep.silent_wrong_tool = ep.outcome == "wrong_answer" and ep.off_plan_calls > 0
    ep.wall_s = time.perf_counter() - t0
    return ep


def _assistant_message(comp: Completion) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": comp.content}
    if comp.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in comp.tool_calls
        ]
    return msg


# ---------------------------------------------------------------------------
# Scripted model
# ---------------------------------------------------------------------------


@dataclass
class LatencyModel:
    """Simulated model latency: fixed + prefill per 1k input tokens + decode per output token."""

    fixed_s: float = 0.35
    per_1k_input_s: float = 0.02
    per_output_s: float = 0.01

    def __call__(self, input_tokens: int, output_tokens: int) -> float:
        return (
            self.fixed_s
            + self.per_1k_input_s * input_tokens / 1000
            + self.per_output_s * output_tokens
        )


def _parse_search_names(text: str) -> list[str]:
    """Tool names from any search output format, in rank order."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d["name"] for d in data if isinstance(d, dict) and "name" in d]
    except (json.JSONDecodeError, TypeError):
        pass
    names = re.findall(r"^### (\S+)", text, flags=re.MULTILINE)
    if names:
        return names
    return re.findall(r"^- (\w+)", text, flags=re.MULTILINE)


class ScriptedModel:
    """Deterministic stand-in for an LLM. It is not a model of LLM behaviour.

    Policy: it knows the reference plan of each task. It selects tools
    perfectly when it can see them (direct / oracle). When it must search,
    it trusts the top-ranked hit, and after an error it tries the next hit
    (at most ``max_candidates``). So with this model, every difference
    between conditions comes from what the server exposes and how discovery
    ranks, not from model reasoning. ``swap_prob`` adds seeded selection noise
    to exercise pass^k and variance reporting.
    """

    def __init__(
        self,
        latency: LatencyModel | None = None,
        max_candidates: int = 3,
        swap_prob: float = 0.0,
        seed: int = 0,
    ):
        self.name = "scripted" + (f"-noise{swap_prob}" if swap_prob else "")
        self.latency = latency or LatencyModel()
        self.max_candidates = max_candidates
        self.swap_prob = swap_prob
        self.seed = seed

    async def start(self, task: Task, condition: str) -> None:
        import random

        self.task = task
        self.condition = condition
        self.rng = random.Random(f"{self.seed}:{task.id}:{condition}")
        self.candidates: dict[str, list[str]] = {}  # capability -> ranked names
        self.bad: set[str] = set()
        self.resolved: dict[str, str] = {}
        self.pending: list[Completion] = []
        self.step = 0
        self._ids = 0
        # direct / search: run the plan in-process, one tool call per turn
        self._req: asyncio.Queue = asyncio.Queue()
        self._runner: asyncio.Task | None = None
        self._awaiting: asyncio.Future | None = None
        self._phase = "start"

    # -- helpers -----------------------------------------------------------
    def _id(self) -> str:
        self._ids += 1
        return f"call_{self._ids}"

    def _call(
        self, name: str, args: dict, tools: list[dict], messages: list[dict]
    ) -> Completion:
        c = Completion(tool_calls=[ToolCall(self._id(), name, args)])
        return self._timed(c, messages, tools)

    def _answer(self, text: str, tools: list[dict], messages: list[dict]) -> Completion:
        return self._timed(Completion(content=text), messages, tools)

    def _timed(
        self, c: Completion, messages: list[dict], tools: list[dict]
    ) -> Completion:
        inp = count_call_tokens(SYSTEM_PROMPT, messages, tools)["input"]
        out = ntok({k: v for k, v in _assistant_message(c).items() if k != "role"})
        c.simulated_latency_s = self.latency(inp, out)
        return c

    def _pick(self, cap: str) -> str | None:
        ranked = [n for n in self.candidates.get(cap, []) if n not in self.bad][
            : self.max_candidates
        ]
        if not ranked:
            return None
        if self.swap_prob and len(ranked) > 1 and self.rng.random() < self.swap_prob:
            return ranked[1]
        return ranked[0]

    def _last_tool_text(self, messages: list[dict]) -> str:
        return (
            messages[-1]["content"]
            if messages and messages[-1]["role"] == "tool"
            else ""
        )

    # -- policy ------------------------------------------------------------
    async def complete(
        self, system: str, messages: list[dict], tools: list[dict]
    ) -> Completion:
        names = {t["function"]["name"] for t in tools}
        if self.condition == "code_mode" or "execute" in names:
            return self._code_mode(messages, tools)
        return await self._sequential(messages, tools, names)

    # Direct / oracle / search: drive the plan coroutine one call at a time.
    async def _sequential(
        self, messages: list[dict], tools: list[dict], names: set[str]
    ) -> Completion:
        searching = "search_tools" in names
        last = messages[-1]
        if self._runner is None:
            self._runner = asyncio.ensure_future(self._run_plan())
        elif self._phase == "await_search":
            cap = self._search_cap
            self.candidates[cap] = [
                n
                for n in _parse_search_names(last["content"])
                if n not in ("search_tools", "call_tool")
            ]
            pick = self._pick(cap)
            if pick is None:
                self._awaiting.set_exception(RuntimeError(f"no tool found for {cap}"))
            else:
                self.resolved[cap] = pick
                return self._issue(pick, self._cur_args, searching, tools, messages)
        elif self._phase == "await_call":
            text = last["content"]
            err = self._last_was_error(text)
            if err and searching:
                self.bad.add(self.resolved[self._cur_cap])
                pick = self._pick(self._cur_cap)
                if pick is not None:
                    self.resolved[self._cur_cap] = pick
                    return self._issue(pick, self._cur_args, searching, tools, messages)
            if err:
                self._awaiting.set_exception(RuntimeError(text))
            else:
                try:
                    self._awaiting.set_result(json.loads(text))
                except json.JSONDecodeError:
                    self._awaiting.set_exception(
                        RuntimeError(f"unparseable result: {text[:80]}")
                    )
        # wait for the plan to ask for the next call or to finish
        get = asyncio.ensure_future(self._req.get())
        done, _ = await asyncio.wait(
            {get, self._runner}, return_when=asyncio.FIRST_COMPLETED
        )
        if get in done:
            cap, args, fut = get.result()
            self._awaiting, self._cur_cap, self._cur_args = fut, cap, args
            if searching and cap not in self.resolved:
                self._phase, self._search_cap = "await_search", cap
                return self._call(
                    "search_tools",
                    {"query": SEARCH_QUERY[cap]}
                    if "query" in self._search_params(tools)
                    else {"pattern": SEARCH_REGEX[cap]},
                    tools,
                    messages,
                )
            name = self.resolved.get(cap, cap)
            return self._issue(name, args, searching, tools, messages)
        get.cancel()
        exc = self._runner.exception()
        if exc is not None:
            return self._answer(f"UNABLE: {exc}"[:200], tools, messages)
        return self._answer(str(self._runner.result()), tools, messages)

    def _issue(
        self,
        name: str,
        args: dict,
        searching: bool,
        tools: list[dict],
        messages: list[dict],
    ) -> Completion:
        self._phase = "await_call"
        if searching:
            return self._call(
                "call_tool", {"name": name, "arguments": args}, tools, messages
            )
        return self._call(name, args, tools, messages)

    @staticmethod
    def _search_params(tools: list[dict]) -> set[str]:
        for t in tools:
            if t["function"]["name"] == "search_tools":
                return set(t["function"]["parameters"].get("properties", {}))
        return set()

    @staticmethod
    def _last_was_error(text: str) -> bool:
        # The runner passes text only; errors from FastMCP start with "Error" or name the failure.
        return (
            text.startswith("Error")
            or "validation error" in text.lower()
            or "Unknown tool" in text
            or "not found" in text
        )

    async def _run_plan(self):
        body = re.sub(TOOL_REF, lambda m: repr(m.group(1)), self.task.solution)
        src = "async def __plan(call_tool):\n" + "\n".join(
            "    " + line for line in body.splitlines()
        )
        ns: dict = {}
        exec(src, ns)  # trusted reference plan from tasks.py

        async def call_tool(cap: str, args: dict):
            fut = asyncio.get_running_loop().create_future()
            await self._req.put((cap, args, fut))
            return await fut

        return await ns["__plan"](call_tool)

    # Code Mode: search each capability, get_schema once, then one execute block.
    def _code_mode(self, messages: list[dict], tools: list[dict]) -> Completion:
        caps = list(self.task.needed)
        last = self._last_tool_text(messages)
        if self._phase == "start":
            self._phase, self._i = "search", 0
        elif self._phase == "search":
            self.candidates[caps[self._i]] = _parse_search_names(last)
            self._i += 1
        elif self._phase == "schema":
            pass
        elif self._phase == "execute":
            m = re.search(r"call_tool\('(\w+)'\) failed|Unknown tool: '?(\w+)", last)
            if m and self._retries < self.max_candidates:
                failed = m.group(1) or m.group(2)
                self.bad.add(failed)
                cap = next((c for c, n in self.resolved.items() if n == failed), None)
                pick = self._pick(cap) if cap else None
                if pick is not None:
                    self.resolved[cap] = pick
                    self._retries += 1
                    self._phase = "schema"
                    return self._call("get_schema", {"tools": [pick]}, tools, messages)
            if last.startswith("Error") or "failed" in last[:200]:
                return self._answer(f"UNABLE: {last[:150]}", tools, messages)
            return self._answer(last.strip(), tools, messages)

        if self._phase == "search" and self._i < len(caps):
            return self._call(
                "search", {"query": SEARCH_QUERY[caps[self._i]]}, tools, messages
            )
        if self._phase == "search":
            for cap in caps:
                pick = self._pick(cap)
                if pick is None:
                    return self._answer(
                        f"UNABLE: no tool found for {cap}", tools, messages
                    )
                self.resolved[cap] = pick
            self._phase, self._retries = "schema", 0
            return self._call(
                "get_schema",
                {"tools": list(dict.fromkeys(self.resolved.values()))},
                tools,
                messages,
            )
        # schema -> execute
        code = re.sub(
            TOOL_REF, lambda m: repr(self.resolved[m.group(1)]), self.task.solution
        )
        self._phase = "execute"
        return self._call("execute", {"code": code}, tools, messages)


def episode_dict(ep: Episode) -> dict:
    d = asdict(ep)
    return d
