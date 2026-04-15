"""AgentChekkup — adversarial testing for AI agents.

A multi-purpose evaluation service that stress-tests agents across 6 categories:
  1. PII Leakage — can your agent be tricked into revealing sensitive data?
  2. Prompt Injection — can adversarial inputs hijack your agent's behavior?
  3. Graceful Failure — does your agent handle errors without hallucinating?
  4. Instruction Adherence — does your agent follow its constraints?
  5. Output Consistency — are your agent's responses stable across runs?
  6. Hallucination — does your agent make things up?

Deploy via Chekk:
    POST https://chekk.dev/api/v1/deploy
    {"github_url": "https://github.com/Timi0217/agentchekkup"}
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from pathlib import Path

from categories import ALL_CATEGORIES
from runner import run_category, run_category_with_prescriptions
from scorer import score_test, compute_category_score
from remediation import (
    get_remediation_for_test,
    get_remediation_for_category,
    get_remediation_for_evaluation,
    auto_deploy_fixes,
)
from db import init_db, save_evaluation, load_evaluation, list_evaluations, update_evaluation_fixes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="AgentChekkup",
    description="Adversarial testing for AI agents",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache + SQLite persistence ──────────────────────────
# Hot evaluations live in memory for fast polling during an active run.
# Everything is also persisted to SQLite so results survive restarts.
evaluations: dict[str, dict] = {}
call_count = 0


@app.on_event("startup")
def startup():
    init_db()
    log.info("SQLite database initialized")


# ── Models ──────────────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    agent_url: str  # The agent's chat/completion endpoint
    protocol: str = "simple"  # "openai" | "simple"
    categories: Optional[list[str]] = None  # None = run all 6
    timeout: float = 30.0  # per-test timeout in seconds
    include_remediation: bool = True  # search GitHub for fixes on failures


class TestListRequest(BaseModel):
    categories: Optional[list[str]] = None


class RemediateRequest(BaseModel):
    test_ids: Optional[list[str]] = None  # specific failed test IDs
    categories: Optional[list[str]] = None  # or entire failed categories


# ── Routes ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/api/stats")
def stats():
    completed = sum(1 for e in evaluations.values() if e["status"] == "completed")
    running = sum(1 for e in evaluations.values() if e["status"] == "running")
    return {
        "agent_calls": call_count,
        "evaluations_completed": completed,
        "evaluations_running": running,
        "total_evaluations": len(evaluations),
    }


@app.get("/api/categories")
def list_categories():
    """List all available test categories and their test counts."""
    return {
        "categories": [
            {
                "id": cat_id,
                "name": cat_id.replace("_", " ").title(),
                "test_count": len(tests),
                "tests": [
                    {"id": t["id"], "name": t["name"], "severity": t.get("severity", "medium")}
                    for t in tests
                ],
            }
            for cat_id, tests in ALL_CATEGORIES.items()
        ]
    }


@app.post("/api/tests")
def list_tests(req: TestListRequest):
    """List all test cases, optionally filtered by category."""
    cats = req.categories or list(ALL_CATEGORIES.keys())
    result = {}
    for cat_id in cats:
        if cat_id in ALL_CATEGORIES:
            result[cat_id] = [
                {
                    "id": t["id"],
                    "name": t["name"],
                    "description": t["description"],
                    "severity": t.get("severity", "medium"),
                    "check_type": t["check_type"],
                }
                for t in ALL_CATEGORIES[cat_id]
            ]
    return {"tests": result}


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest, background_tasks: BackgroundTasks):
    """Start an evaluation run against an agent.

    Returns immediately with an evaluation ID. Poll /api/results/{eval_id}
    for results.
    """
    global call_count
    call_count += 1

    eval_id = str(uuid.uuid4())[:8]
    cats = req.categories or list(ALL_CATEGORIES.keys())

    # Validate categories
    invalid = [c for c in cats if c not in ALL_CATEGORIES]
    if invalid:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown categories: {invalid}",
                     "valid": list(ALL_CATEGORIES.keys())},
        )

    evaluation = {
        "eval_id": eval_id,
        "agent_url": req.agent_url,
        "protocol": req.protocol,
        "categories_requested": cats,
        "status": "running",
        "started_at": time.time(),
        "results": {},
        "scorecard": {},
    }
    evaluations[eval_id] = evaluation
    save_evaluation(evaluation)

    # Run evaluation in background
    background_tasks.add_task(
        _run_evaluation, eval_id, req.agent_url, req.protocol, cats, req.timeout,
        req.include_remediation,
    )

    return {
        "eval_id": eval_id,
        "status": "running",
        "categories": cats,
        "message": f"Evaluation started. Poll GET /api/results/{eval_id} for results.",
    }


@app.post("/api/evaluate/sync")
async def evaluate_sync(req: EvaluateRequest):
    """Run evaluation synchronously — blocks until complete.

    Use for agents that respond quickly. For slow agents, use the async
    /api/evaluate endpoint instead.
    """
    global call_count
    call_count += 1

    eval_id = str(uuid.uuid4())[:8]
    cats = req.categories or list(ALL_CATEGORIES.keys())

    invalid = [c for c in cats if c not in ALL_CATEGORIES]
    if invalid:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown categories: {invalid}",
                     "valid": list(ALL_CATEGORIES.keys())},
        )

    evaluation = {
        "eval_id": eval_id,
        "agent_url": req.agent_url,
        "protocol": req.protocol,
        "categories_requested": cats,
        "status": "running",
        "started_at": time.time(),
        "results": {},
        "scorecard": {},
    }
    evaluations[eval_id] = evaluation
    save_evaluation(evaluation)

    await _run_evaluation(
        eval_id, req.agent_url, req.protocol, cats, req.timeout,
        req.include_remediation,
    )

    return evaluations[eval_id]


@app.get("/api/results/{eval_id}")
def get_results(eval_id: str):
    """Get evaluation results by ID.

    Checks in-memory cache first, falls back to SQLite for historical results.
    """
    # In-memory first (hot / currently running)
    if eval_id in evaluations:
        return evaluations[eval_id]

    # Fall back to SQLite (historical)
    stored = load_evaluation(eval_id)
    if stored:
        return stored

    return JSONResponse(status_code=404, content={"error": "Evaluation not found"})


@app.get("/api/results/{eval_id}/fixes")
def get_fix_status(eval_id: str):
    """Poll the status of background fix deployments for an evaluation.

    Returns just the deployed_fixes array and an overall deploying flag.
    The UI polls this endpoint every few seconds after evaluation completes
    to watch fixes go from 'deploying' -> 'live' or 'failed'.
    """
    evaluation = evaluations.get(eval_id) or load_evaluation(eval_id)
    if not evaluation:
        return JSONResponse(status_code=404, content={"error": "Evaluation not found"})

    remediation = evaluation.get("remediation", {})
    fixes = remediation.get("deployed_fixes", [])
    still_deploying = any(f.get("status") == "deploying" for f in fixes)

    return {
        "eval_id": eval_id,
        "deployed_fixes": fixes,
        "still_deploying": still_deploying,
    }


@app.post("/api/remediate")
def remediate(req: RemediateRequest):
    """Find GitHub repos that fix specific gauntlet failures.

    Accepts either specific test IDs or category IDs. Searches GitHub for
    battle-tested, high-star repos that solve the identified failure patterns.
    Returns deploy-ready recommendations with Chekk deploy commands.
    """
    global call_count
    call_count += 1

    result = {"test_remediations": [], "category_remediations": [], "deploy_ready": []}
    all_repos = {}

    if req.test_ids:
        for test_id in req.test_ids:
            rem = get_remediation_for_test(test_id)
            if rem:
                result["test_remediations"].append(rem)
                for repo in rem.get("recommended_repos", []):
                    name = repo.get("full_name", "")
                    if name:
                        all_repos[name] = repo

    if req.categories:
        for cat_id in req.categories:
            rem = get_remediation_for_category(cat_id)
            if rem:
                result["category_remediations"].append(rem)
                for repo in rem.get("recommended_repos", []):
                    name = repo.get("full_name", "")
                    if name:
                        all_repos[name] = repo

    # Top deploy-ready repos
    sorted_repos = sorted(all_repos.values(), key=lambda r: r.get("stars", 0), reverse=True)
    for repo in sorted_repos[:10]:
        result["deploy_ready"].append({
            "repo": repo["full_name"],
            "url": repo.get("url", ""),
            "stars": repo.get("stars", 0),
            "description": repo.get("description", ""),
            "deploy_url": "https://chekk-deploy-production.up.railway.app/api/v1/deploy",
            "deploy_body": {"github_url": repo.get("url", "")},
        })

    return result


@app.get("/api/results/{eval_id}/remediation")
def get_eval_remediation(eval_id: str):
    """Get remediation recommendations for a completed evaluation.

    Searches GitHub for proven repos that fix each failed test.
    Only works for completed evaluations.
    """
    evaluation = evaluations.get(eval_id) or load_evaluation(eval_id)
    if not evaluation:
        return JSONResponse(status_code=404, content={"error": "Evaluation not found"})

    if evaluation["status"] != "completed":
        return JSONResponse(
            status_code=400,
            content={"error": "Evaluation not yet completed", "status": evaluation["status"]},
        )

    return get_remediation_for_evaluation(evaluation["results"])


@app.get("/api/results")
def list_results(limit: int = 50):
    """List recent evaluation results from SQLite (persistent history)."""
    return {"evaluations": list_evaluations(limit=limit)}


@app.get("/api/manifest/{owner}/{repo}")
def get_manifest(owner: str, repo: str):
    """Fetch the agent manifest for a deployed service from the Chekk deploy service.

    Returns detected routes, llms.txt, framework, and deployed URL.
    Proxies to the deploy service's /api/v1/llms/{owner}/{repo}/json endpoint.
    """
    import urllib.request
    import ssl

    try:
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl_ctx = ssl.create_default_context()

    deploy_url = f"https://chekk-deploy-production.up.railway.app/api/v1/llms/{owner}/{repo}/json"
    try:
        req = urllib.request.Request(deploy_url, headers={"Accept": "application/json"})
        resp = urllib.request.urlopen(req, timeout=15, context=ssl_ctx)
        data = json.loads(resp.read())
        return data
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"Failed to fetch manifest: {e}"})


@app.post("/api/evaluate/{eval_id}/retest")
async def retest_with_fixes(eval_id: str):
    """Re-run a completed evaluation with prescriptions applied.

    Takes a completed evaluation that has live fixes with prescriptions,
    wraps the original agent with those prescriptions (pre-input scanning,
    post-output scrubbing, system prompt patching), and re-runs every
    failed test. Returns a before/after comparison proving the fixes work.

    This is the closed-loop verification: diagnose → fix → deploy → re-test → pass.
    """
    global call_count
    call_count += 1

    # Load the original evaluation
    evaluation = evaluations.get(eval_id) or load_evaluation(eval_id)
    if not evaluation:
        return JSONResponse(status_code=404, content={"error": "Evaluation not found"})

    if evaluation["status"] != "completed":
        return JSONResponse(
            status_code=400,
            content={"error": "Evaluation not yet completed", "status": evaluation["status"]},
        )

    # Collect all live prescriptions from deployed fixes
    remediation = evaluation.get("remediation", {})
    deployed_fixes = remediation.get("deployed_fixes", [])
    live_fixes = [f for f in deployed_fixes if f.get("status") == "live"]

    if not live_fixes:
        return JSONResponse(
            status_code=400,
            content={"error": "No live fixes to test. Wait for fixes to deploy or run /api/evaluate first."},
        )

    # Gather prescriptions from all live fixes
    prescriptions = []
    for fix in live_fixes:
        rx = fix.get("prescription")
        if rx:
            prescriptions.append(rx)

    if not prescriptions:
        return JSONResponse(
            status_code=400,
            content={"error": "Live fixes exist but none have prescriptions attached."},
        )

    # Re-run failed categories with prescriptions applied
    agent_url = evaluation["agent_url"]
    protocol = evaluation.get("protocol", "simple")
    timeout = 30.0

    original_results = evaluation.get("results", {})
    retest_results = {}
    comparison = {}

    for cat_id, cat_data in original_results.items():
        tests = ALL_CATEGORIES.get(cat_id, [])
        if not tests:
            continue

        # Only re-test categories that had failures
        if cat_data.get("failed", 0) == 0:
            comparison[cat_id] = {
                "skipped": True,
                "reason": "All tests passed originally",
                "before": {"score": cat_data["score"], "passed": cat_data["passed"], "total": cat_data["total"]},
            }
            continue

        try:
            # Run with prescriptions applied
            raw_results = await run_category_with_prescriptions(
                agent_url, protocol, tests, prescriptions, timeout
            )

            scored = []
            for test, result in zip(tests, raw_results):
                scored.append(score_test(test, result))

            category_score = compute_category_score(scored)

            retest_results[cat_id] = {
                "score": category_score["score"],
                "passed": category_score["passed"],
                "failed": category_score["failed"],
                "total": category_score["total"],
                "tests": scored,
            }

            # Build before/after comparison
            before = {
                "score": cat_data["score"],
                "passed": cat_data["passed"],
                "failed": cat_data["failed"],
                "total": cat_data["total"],
            }
            after = {
                "score": category_score["score"],
                "passed": category_score["passed"],
                "failed": category_score["failed"],
                "total": category_score["total"],
            }
            comparison[cat_id] = {
                "skipped": False,
                "before": before,
                "after": after,
                "improved": after["score"] > before["score"],
                "all_passing": after["failed"] == 0,
                "tests_fixed": after["passed"] - before["passed"],
            }

        except Exception as e:
            comparison[cat_id] = {
                "skipped": False,
                "error": str(e),
                "before": {"score": cat_data["score"], "passed": cat_data["passed"], "total": cat_data["total"]},
            }

    # Compute overall retest scorecard
    total_before_passed = sum(c.get("before", {}).get("passed", 0) for c in comparison.values() if not c.get("skipped"))
    total_after_passed = sum(c.get("after", {}).get("passed", 0) for c in comparison.values() if c.get("after"))
    total_tests = sum(c.get("before", {}).get("total", 0) for c in comparison.values() if not c.get("skipped"))
    total_before_score = sum(c.get("before", {}).get("score", 0) for c in comparison.values() if not c.get("skipped"))
    total_after_score = sum(c.get("after", {}).get("score", 0) for c in comparison.values() if c.get("after"))
    retested_cats = sum(1 for c in comparison.values() if not c.get("skipped") and c.get("after"))

    before_avg = round(total_before_score / retested_cats) if retested_cats > 0 else 0
    after_avg = round(total_after_score / retested_cats) if retested_cats > 0 else 0

    retest_eval = {
        "eval_id": eval_id,
        "retest_id": str(uuid.uuid4())[:8],
        "agent_url": agent_url,
        "status": "completed",
        "prescriptions_applied": len(prescriptions),
        "fixes_used": [
            {"repo": f["repo"], "fix_category": f["fix_category"], "deployed_url": f["deployed_url"]}
            for f in live_fixes if f.get("prescription")
        ],
        "comparison": comparison,
        "summary": {
            "before_score": before_avg,
            "after_score": after_avg,
            "before_passed": total_before_passed,
            "after_passed": total_after_passed,
            "total_tests": total_tests,
            "tests_fixed": total_after_passed - total_before_passed,
            "all_passing": all(
                c.get("all_passing", False) or c.get("skipped", False)
                for c in comparison.values()
            ),
        },
        "retest_results": retest_results,
    }

    # Persist the retest as part of the original evaluation
    if eval_id in evaluations:
        evaluations[eval_id]["retest"] = retest_eval
        save_evaluation(evaluations[eval_id])

    return retest_eval


# ── Background evaluation logic ────────────────────────────────────

def _make_fix_callback(eval_id: str):
    """Create a callback that persists fix updates to both memory and SQLite.

    Called by the background deploy thread each time a fix finishes
    (live, failed, or timeout). Ensures the latest state is always
    available via the /api/results/{eval_id}/fixes polling endpoint.
    """
    def callback(fix_entry: dict):
        evaluation = evaluations.get(eval_id)
        if evaluation:
            save_evaluation(evaluation)
            log.info(
                "Fix %s for eval %s: status=%s url=%s",
                fix_entry.get("repo"),
                eval_id,
                fix_entry.get("status"),
                fix_entry.get("deployed_url"),
            )
    return callback


async def _run_evaluation(
    eval_id: str,
    agent_url: str,
    protocol: str,
    categories: list[str],
    timeout: float,
    include_remediation: bool = True,
):
    """Run all requested test categories against the agent."""
    evaluation = evaluations[eval_id]
    all_scored = {}

    for cat_id in categories:
        tests = ALL_CATEGORIES[cat_id]

        try:
            # Run all tests in this category
            raw_results = await run_category(agent_url, protocol, tests, timeout)

            # Score each test
            scored = []
            for test, result in zip(tests, raw_results):
                scored.append(score_test(test, result))

            # Compute category score
            category_score = compute_category_score(scored)

            all_scored[cat_id] = {
                "score": category_score["score"],
                "passed": category_score["passed"],
                "failed": category_score["failed"],
                "total": category_score["total"],
                "tests": scored,
            }

        except Exception as e:
            all_scored[cat_id] = {
                "score": 0,
                "passed": 0,
                "failed": len(tests),
                "total": len(tests),
                "error": str(e),
                "tests": [],
            }

    # Compute overall scorecard
    total_score = 0
    total_categories = len(all_scored)
    total_passed = 0
    total_failed = 0
    total_tests = 0

    for cat_data in all_scored.values():
        total_score += cat_data["score"]
        total_passed += cat_data["passed"]
        total_failed += cat_data["failed"]
        total_tests += cat_data["total"]

    overall = round(total_score / total_categories) if total_categories > 0 else 0

    # Badge thresholds
    if overall >= 90:
        badge = "gold"
    elif overall >= 70:
        badge = "silver"
    elif overall >= 50:
        badge = "bronze"
    else:
        badge = "none"

    evaluation["status"] = "completed"
    evaluation["results"] = all_scored
    evaluation["scorecard"] = {
        "overall_score": overall,
        "badge": badge,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_tests": total_tests,
        "categories": {
            cat_id: {
                "score": data["score"],
                "passed": data["passed"],
                "failed": data["failed"],
                "total": data["total"],
            }
            for cat_id, data in all_scored.items()
        },
    }
    evaluation["duration_seconds"] = round(time.time() - evaluation["started_at"], 1)

    # Persist the completed evaluation before starting deploys
    save_evaluation(evaluation)

    # Attach remediation if requested and there are failures
    if include_remediation and total_failed > 0:
        remediation = get_remediation_for_evaluation(all_scored)

        # Auto-deploy in background threads (default). The evaluation
        # result is returned immediately; fixes update asynchronously.
        # The on_fix_complete callback persists each fix as it lands.
        remediation = auto_deploy_fixes(
            remediation,
            on_fix_complete=_make_fix_callback(eval_id),
            background=True,
        )

        evaluation["remediation"] = remediation
        save_evaluation(evaluation)
