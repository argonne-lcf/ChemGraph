"""Benchmark ALCF IRI skill card through headless Claude Code.

Drives Claude Code (`claude -p`) against each question in the qeval
notebook. Loads src/chemgraph/skills/alcf_iri_bash.md as a system-prompt
append so the coding agent knows how to hit the IRI API via curl.
Writes one JSONL row per (question, trial): question + final answer +
full tool-call transcript + wall time.

Feed the resulting JSONL into iri_qeval.ipynb's binary judge cell to
score correctness. See how-to-run comments at bottom.

Prereqs:
  * `claude` CLI on PATH (2.x)
  * ALCF_API_TOKEN in env, OR ~/.globus/app/8b84fc2d-.../alcf_facility_api_app/tokens.json
    populated (skill's Auth section handles the second path)
  * curl + jq available in the shell Claude Code spawns
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = REPO_ROOT / "src" / "chemgraph" / "skills" / "alcf_iri_bash.md"

# Same 15 questions as iri_qeval.ipynb (kept in sync manually; the notebook
# is the source of truth for wording).
QUESTIONS: list[tuple[str, str]] = [
    ("q1",  "How many total resources does ALCF facility API list?"),
    ("q2",  "How many compute resources are currently marked \"up\"?"),
    ("q3",  "What is the UUID of the resource named \"Aurora\"?"),
    ("q4",  "How many storage resources are listed (any status)?"),
    ("q5",  "How many capabilities does the facility expose?"),
    ("q6",  "What is the internal id of the capability named \"aurora\"?"),
    ("q7",  "How many projects does my ALCF API token grant me access to?"),
    ("q8",  "How many user_ids are on the ChemGraph project?"),
    ("q9",  "How many allocations does the ChemGraph project have across all machines?"),
    ("q10", "What is the total node_hours allocated to ChemGraph across all machines?"),
    ("q11", "Which of ChemGraph's allocations has the highest usage-to-allocation ratio? Answer with the capability name."),
    ("q12", "How many total jobs are currently in Polaris's active queue (across all users, all states)?"),
    ("q13", "How many jobs on Polaris are in state \"queued\" (not held, not active)?"),
    ("q14", "How many jobs on Crux are in state \"active\"?"),
    ("q15", "What is the numeric PBS id of the oldest queued Polaris job?"),
    ("q20", "How many tasks does the /task queue currently contain for my token?"),
]

# The skill is loaded via --append-system-prompt-file so Claude Code
# has it in context for every turn. Only Bash is enabled -- no Read,
# Write, WebFetch etc. That keeps the runtime honest: it must curl its
# way to the answer, not read local caches or invent URLs.
#
# Bash isn't tighter-scoped than that because the skill uses composed
# shell (pipes, subshells, PID=$(curl ...)) and prefix-based Bash
# ACLs interact badly with $(...). If you want stricter isolation,
# run this in a container or a dedicated user.
ALLOWED_TOOLS = ["Bash"]

# Per-question dollar cap. Belt-and-suspenders vs. the agent
# looping on a bad request. Adjust upward if you see truncations.
MAX_BUDGET_USD = 1.50


def build_claude_cmd(question: str, skill_body: str,
                      bare: bool = True) -> list[str]:
    """Build the argv to launch one headless Claude Code subprocess.

    ``bare=True`` isolates the run from ambient project state (memory,
    CLAUDE.md, hooks, plugins) so the ONLY IRI knowledge Claude Code
    has is what the appended skill teaches it. That's the fair test.
    Set False if you deliberately want to measure skill+ambient.
    """
    cmd = ["claude", "-p"]
    if bare:
        cmd.append("--bare")
    cmd += [
        "--output-format", "json",
        "--allowed-tools", *ALLOWED_TOOLS,
        # Append the entire skill as extra system context. The CLI's
        # --append-system-prompt takes a raw string (no -file variant in
        # 2.1.x), so we pass the file body directly.
        "--append-system-prompt", skill_body,
        "--max-budget-usd", str(MAX_BUDGET_USD),
        "--allow-dangerously-skip-permissions",  # no tty prompts; we already restricted tools
        question,
    ]
    return cmd


async def run_one(qid: str, question: str, trial: int,
                   skill_body: str, *,
                   timeout_s: int = 240,
                   bare: bool = True) -> dict:
    t0 = time.perf_counter()
    cmd = build_claude_cmd(question, skill_body, bare=bare)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            # Explicitly close stdin. Without this, `claude -p` waits
            # up to 3s for piped input, then warns and exits 1 without
            # processing the prompt -- silently killing the benchmark.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return _row(qid, question, trial, t0,
                        ok=False, error=f"timeout after {timeout_s}s")
    except FileNotFoundError as e:
        return _row(qid, question, trial, t0, ok=False, error=repr(e))

    wall_ms = int((time.perf_counter() - t0) * 1000)

    if proc.returncode != 0:
        return _row(qid, question, trial, t0, ok=False,
                    error=f"claude exit {proc.returncode}: {stderr.decode()[:500]}",
                    wall_ms=wall_ms)

    try:
        parsed = json.loads(stdout.decode())
    except Exception as e:
        return _row(qid, question, trial, t0, ok=False,
                    error=f"stdout parse failed: {e!r}",
                    wall_ms=wall_ms,
                    raw_stdout=stdout.decode()[:2000])

    # Claude Code's --output-format json returns a dict with 'result'
    # (final assistant text), 'messages' (full turn log incl. tool
    # calls), 'total_cost_usd', 'usage', etc. Extract what we need.
    answer = parsed.get("result", "")
    trace = _render_trace(parsed.get("messages", []))
    usage = parsed.get("usage", {}) or {}

    # Claude Code can exit 0 with is_error=true (auth failed, tool
    # denied, etc.) -- treat that as a failure so the row surfaces
    # the real cause instead of ok=True with an empty trace.
    cc_is_error = bool(parsed.get("is_error"))

    return _row(
        qid, question, trial, t0,
        ok=(not cc_is_error),
        error=(answer if cc_is_error else None),
        wall_ms=wall_ms,
        answer=answer,
        trace_rendered=trace,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=(usage.get("input_tokens", 0)
                       + usage.get("output_tokens", 0)),
        turns=parsed.get("num_turns", 0),
        cost_usd=parsed.get("total_cost_usd"),
    )


def _render_trace(messages: list) -> str:
    """Flatten Claude Code's messages array into a compact text trace
    the binary judge can read. Same shape as the notebook's
    _trace_from_state output: CALL <tool>(args) / RESULT[<tool>]: ..."""
    lines = []
    for m in messages:
        # Claude Code emits messages in the Anthropic API shape:
        # {role: 'assistant', content: [{type: 'tool_use', name, input}, ...]}
        # {role: 'user',      content: [{type: 'tool_result', tool_use_id, content}, ...]}
        if not isinstance(m, dict):
            continue
        content = m.get("content") or m.get("message", {}).get("content", [])
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t == "tool_use":
                name = part.get("name", "?")
                inp = json.dumps(part.get("input", {}), default=str)
                lines.append(f"CALL {name}({inp[:400]})")
            elif t == "tool_result":
                inner = part.get("content", "")
                if isinstance(inner, list):
                    inner = "".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in inner
                    )
                inner = str(inner).replace("\n", " ")
                if len(inner) > 4000:
                    inner = inner[:4000] + f" ... (truncated, {len(inner)} total chars)"
                lines.append(f"RESULT: {inner}")
    return "\n".join(lines) if lines else "(no tool calls)"


def _row(qid, question, trial, t0, **extra) -> dict:
    base = dict(
        runtime="claude_code",
        skill="alcf_iri_bash.md",
        qid=qid,
        question=question,
        trial=trial,
        answer="",
        trace_rendered="",
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        turns=0,
        cost_usd=None,
        wall_ms=int((time.perf_counter() - t0) * 1000),
        ok=True,
        error=None,
    )
    base.update(extra)
    return base


async def run_all(*, qids: list[str] | None = None,
                    trials: int = 1,
                    concurrency: int = 2,
                    bare: bool = True,
                    out_path: Path | None = None,
                    on_progress=None) -> list[dict]:
    """Async entrypoint usable from notebooks and scripts.

    Runs the selected questions x trials, writes an optional JSONL,
    and returns the list of result rows. Suitable for
    ``await run_all(...)`` inside a Jupyter cell.

    Parameters
    ----------
    qids : list of str, optional
        Subset of question IDs (e.g. ["q1", "q2"]). Default: all 15.
    trials : int
        Runs per question. Default 1.
    concurrency : int
        Parallel claude subprocesses. Default 2. Claude Code isn't
        cheap; raise cautiously.
    bare : bool
        If True (default), pass ``--bare`` so ambient project state
        (memory, CLAUDE.md, hooks) can't influence the run.
    out_path : Path, optional
        Write a JSONL to this path if given.
    on_progress : callable(done_count, total, row), optional
        Called after each row finishes. Useful for tqdm/log updates.
    """
    if not SKILL_PATH.exists():
        raise FileNotFoundError(f"skill file not found: {SKILL_PATH}")
    skill_body = SKILL_PATH.read_text()

    qs = QUESTIONS
    if qids:
        keep = set(qids)
        qs = [(q, t) for q, t in QUESTIONS if q in keep]
        if not qs:
            raise ValueError(f"no matching qids from {sorted(keep)}")

    jobs = [(qid, q, trial) for qid, q in qs for trial in range(trials)]
    total = len(jobs)
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def _wrap(job):
        nonlocal done
        async with sem:
            row = await run_one(*job, skill_body=skill_body, bare=bare)
            done += 1
            if on_progress is not None:
                try:
                    on_progress(done, total, row)
                except Exception:
                    pass
            return row

    rows = list(await asyncio.gather(*[_wrap(j) for j in jobs]))

    if out_path is not None:
        with Path(out_path).open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return rows


async def _main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2,
                        help="parallel claude subprocesses (default 2)")
    parser.add_argument("--out", type=Path, default=Path("iri_bench_claude_code.jsonl"))
    parser.add_argument("--qids", nargs="+", default=None,
                        help="subset of qids to run (default: all)")
    parser.add_argument("--no-bare", action="store_true",
                        help="drop --bare so Claude Code reads memory + CLAUDE.md")
    args = parser.parse_args()

    def _log(done, total, row):
        print(f"[{done}/{total}] {row['qid']} trial={row['trial']} "
              f"ok={row['ok']} wall={row['wall_ms']}ms", file=sys.stderr)

    rows = await run_all(
        qids=args.qids,
        trials=args.trials,
        concurrency=args.concurrency,
        bare=not args.no_bare,
        out_path=args.out,
        on_progress=_log,
    )
    print(f"[bench] wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(_main_cli())


# ---------------------------------------------------------------------------
# How to run
# ---------------------------------------------------------------------------
#
# 1. Populate ALCF_API_TOKEN (or ensure the on-disk Globus cache is valid).
#    See src/chemgraph/skills/alcf_iri_bash.md § Auth for the two paths.
#
# 2. Full benchmark (15 questions x 1 trial, ~5-15 min depending on Claude
#    Code latency; costs several USD -- check ANTHROPIC_API_KEY tier):
#
#      cd examples/iri/
#      python bench_claude_code.py --trials 1 --concurrency 2 \
#           --out iri_bench_claude_code.jsonl
#
# 3. One-question sanity check (~30-60s, ~$0.05):
#
#      python bench_claude_code.py --qids q1 --trials 1
#
# 4. Score the JSONL under the notebook's binary judge. Add a cell to
#    iri_qeval.ipynb (after the binary-judge cell) with:
#
#      import json as _j
#      from pathlib import Path
#      cc_rows = [_j.loads(l) for l in Path('iri_bench_claude_code.jsonl').read_text().splitlines()]
#      judge_llm = load_chat_model(model_name=JUDGE_MODEL, temperature=0.0,
#                                  base_url=BASE_URL, argo_user=ARGO_USER)
#      import asyncio
#      async def _score():
#          out = []
#          for r in cc_rows:
#              v = await judge_run_binary(judge_llm, r['question'],
#                                          r['trace_rendered'], r['answer'])
#              out.append({**r, 'judge_binary': v})
#          return out
#      cc_scored = await _score()
#      correct = sum(1 for r in cc_scored if (r.get('judge_binary') or {}).get('score') == 1)
#      print(f'Claude Code + bash skill: {correct}/{len(cc_scored)} correct')
#
# 5. To compare against single_agent_iri under the SAME binary judge,
#    also rescore the notebook's existing rubric-scored `results` list:
#
#      # `results` already in memory from the sweep cell
#      async def _score_iri():
#          out = []
#          for r in results:
#              v = await judge_run_binary(judge_llm, r['question'],
#                                          # rubric sweep drops raw state; use answer alone
#                                          '(trace not preserved)',
#                                          r['answer'])
#              out.append({**r, 'judge_binary': v})
#          return out
#
#    NOTE: the notebook's sweep cell drops `state` (row['state']=None) after
#    judging to keep the JSON small. To get apples-to-apples binary scoring,
#    modify the sweep to keep a rendered `trace_rendered` field alongside
#    the rubric verdict. Cheap one-line addition:
#
#        m['trace_rendered'] = _trace_from_state(m['state'])
#        m.pop('state', None)
#
#    (place before the `return dict(...)` at the end of `_do` in the sweep
#    cell). Re-run the sweep once with that change, then binary scoring
#    works for both runtimes with the same evidence quality.
