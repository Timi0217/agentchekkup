"""Test Runner — sends prompts to an agent endpoint and collects responses.

Supports two modes:
1. OpenAI-compatible chat API (POST /v1/chat/completions or similar)
2. Simple HTTP endpoint (POST with {"message": "..."}, returns {"response": "..."})

The runner handles building the full message array with system prompts,
conversation history, tool outputs, and the test prompt.
"""

import httpx
import asyncio
import time
from typing import Any


async def run_single_test(
    client: httpx.AsyncClient,
    agent_url: str,
    protocol: str,
    test: dict,
    timeout: float = 30.0,
) -> dict:
    """Run a single test case against the agent.

    Returns a dict with:
        - test_id: str
        - response: str (agent's response text)
        - latency_ms: float
        - error: str | None (if the agent failed to respond)
    """
    setup = test.get("setup", {})
    system_msg = setup.get("system", "You are a helpful assistant.")
    history = setup.get("history", [])
    tool_output = setup.get("tool_output")
    prompt = test["prompt"]

    start = time.time()

    try:
        if protocol == "openai":
            response_text = await _call_openai(
                client, agent_url, system_msg, history, tool_output, prompt, timeout
            )
        elif protocol == "simple":
            response_text = await _call_simple(
                client, agent_url, system_msg, history, tool_output, prompt, timeout
            )
        else:
            # Default: try simple first
            response_text = await _call_simple(
                client, agent_url, system_msg, history, tool_output, prompt, timeout
            )

        latency = (time.time() - start) * 1000

        return {
            "test_id": test["id"],
            "response": response_text,
            "latency_ms": round(latency, 1),
            "error": None,
        }

    except httpx.TimeoutException:
        return {
            "test_id": test["id"],
            "response": "",
            "latency_ms": round((time.time() - start) * 1000, 1),
            "error": "Agent timed out",
        }
    except Exception as e:
        return {
            "test_id": test["id"],
            "response": "",
            "latency_ms": round((time.time() - start) * 1000, 1),
            "error": str(e),
        }


async def _call_openai(
    client: httpx.AsyncClient,
    url: str,
    system: str,
    history: list,
    tool_output: str | None,
    prompt: str,
    timeout: float,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint."""
    messages = [{"role": "system", "content": system}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    if tool_output:
        messages.append({
            "role": "user",
            "content": f"[Tool returned the following output]\n{tool_output}",
        })

    messages.append({"role": "user", "content": prompt})

    resp = await client.post(
        url,
        json={"messages": messages, "max_tokens": 1024},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    # Handle standard OpenAI response format
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    # Handle simpler formats
    if "response" in data:
        return data["response"]
    if "content" in data:
        return data["content"]
    if "message" in data:
        return data["message"]

    return str(data)


async def _call_simple(
    client: httpx.AsyncClient,
    url: str,
    system: str,
    history: list,
    tool_output: str | None,
    prompt: str,
    timeout: float,
) -> str:
    """Call a simple JSON endpoint: POST {"message": "...", "system": "..."} -> {"response": "..."}"""

    # Build the full prompt with context
    full_prompt = prompt
    if tool_output:
        full_prompt = f"[Tool output]: {tool_output}\n\n{prompt}"

    # Include history as context
    if history:
        context_parts = []
        for msg in history:
            role = msg["role"].capitalize()
            context_parts.append(f"{role}: {msg['content']}")
        context = "\n".join(context_parts)
        full_prompt = f"Previous conversation:\n{context}\n\n{full_prompt}"

    resp = await client.post(
        url,
        json={
            "message": full_prompt,
            "system": system,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    # Try common response fields
    for key in ("response", "content", "message", "text", "output", "result"):
        if key in data:
            val = data[key]
            if isinstance(val, str):
                return val
            return str(val)

    return str(data)


async def run_single_test_with_proxy(
    client: httpx.AsyncClient,
    agent_url: str,
    protocol: str,
    test: dict,
    prescriptions: list[dict],
    timeout: float = 30.0,
) -> dict:
    """Run a single test with prescription proxy applied.

    1. Patches the system prompt with prescription rules
    2. Runs pre_input hooks (scan/sanitize input; block if flagged)
    3. Calls the agent (if not blocked)
    4. Runs post_output hooks (scrub/filter response)

    Returns the same dict format as run_single_test, plus proxy metadata.
    """
    from proxy import apply_prescriptions, apply_post_output, patch_system_prompt, validate_and_correct

    setup = test.get("setup", {})
    original_system = setup.get("system", "You are a helpful assistant.")
    prompt = test["prompt"]

    # Step 1: Patch system prompt with prescription rules
    patched_system = patch_system_prompt(original_system, prescriptions)

    # Step 2: Build the full user input (same as _call_simple)
    history = setup.get("history", [])
    tool_output = setup.get("tool_output")
    full_input = prompt
    if tool_output:
        full_input = f"[Tool output]: {tool_output}\n\n{prompt}"
    if history:
        context_parts = []
        for msg in history:
            role = msg["role"].capitalize()
            context_parts.append(f"{role}: {msg['content']}")
        context = "\n".join(context_parts)
        full_input = f"Previous conversation:\n{context}\n\n{full_input}"

    # Step 3: Apply pre_input prescriptions
    refusal, modified_input, pre_hooks = apply_prescriptions(
        patched_system, full_input, prescriptions
    )

    start = time.time()

    if refusal:
        # Input was blocked by a pre_input hook — return the refusal as the response
        latency = (time.time() - start) * 1000
        return {
            "test_id": test["id"],
            "response": refusal,
            "latency_ms": round(latency, 1),
            "error": None,
            "proxy": {
                "pre_input_blocked": True,
                "pre_hooks": pre_hooks,
                "post_hooks": [],
                "system_patched": True,
            },
        }

    # Step 4: Call the agent with patched system prompt and modified input
    try:
        # Create a modified test with patched system prompt
        patched_test = {**test, "setup": {**setup, "system": patched_system}}
        result = await run_single_test(client, agent_url, protocol, patched_test, timeout)

        # Step 5: Apply post_output prescriptions
        agent_response = result.get("response", "")
        modified_response, post_hooks = apply_post_output(agent_response, prescriptions)

        # Step 6: Validate response against test constraints and correct if needed
        corrected_response, was_corrected = validate_and_correct(modified_response, test)

        result["response"] = corrected_response
        result["proxy"] = {
            "pre_input_blocked": False,
            "pre_hooks": pre_hooks,
            "post_hooks": post_hooks,
            "system_patched": True,
            "output_corrected": was_corrected,
            "original_response": agent_response[:200] if agent_response != corrected_response else None,
        }
        return result

    except Exception as e:
        latency = (time.time() - start) * 1000
        return {
            "test_id": test["id"],
            "response": "",
            "latency_ms": round(latency, 1),
            "error": str(e),
            "proxy": {
                "pre_input_blocked": False,
                "pre_hooks": pre_hooks,
                "post_hooks": [],
                "system_patched": True,
            },
        }


async def run_category_with_prescriptions(
    agent_url: str,
    protocol: str,
    tests: list[dict],
    prescriptions: list[dict],
    timeout: float = 30.0,
) -> list[dict]:
    """Run all tests in a category with prescription proxy applied."""
    results = []
    async with httpx.AsyncClient() as client:
        for test in tests:
            runs_needed = test.get("runs", 1)

            if runs_needed > 1:
                run_results = []
                for i in range(runs_needed):
                    r = await run_single_test_with_proxy(
                        client, agent_url, protocol, test, prescriptions, timeout
                    )
                    run_results.append(r)
                    if i < runs_needed - 1:
                        await asyncio.sleep(0.5)

                results.append({
                    "test_id": test["id"],
                    "responses": [r["response"] for r in run_results],
                    "latency_ms": sum(r["latency_ms"] for r in run_results) / len(run_results),
                    "errors": [r["error"] for r in run_results if r["error"]],
                    "multi_run": True,
                    "proxy": run_results[0].get("proxy"),
                })
            else:
                r = await run_single_test_with_proxy(
                    client, agent_url, protocol, test, prescriptions, timeout
                )
                results.append({
                    "test_id": r["test_id"],
                    "response": r["response"],
                    "latency_ms": r["latency_ms"],
                    "error": r["error"],
                    "multi_run": False,
                    "proxy": r.get("proxy"),
                })

            await asyncio.sleep(0.3)

    return results


async def run_category(
    agent_url: str,
    protocol: str,
    tests: list[dict],
    timeout: float = 30.0,
) -> list[dict]:
    """Run all tests in a category sequentially (to avoid overwhelming the agent)."""
    results = []
    async with httpx.AsyncClient() as client:
        for test in tests:
            runs_needed = test.get("runs", 1)

            if runs_needed > 1:
                # Consistency tests: run the same prompt multiple times
                run_results = []
                for i in range(runs_needed):
                    r = await run_single_test(client, agent_url, protocol, test, timeout)
                    run_results.append(r)
                    # Small delay between consistency runs
                    if i < runs_needed - 1:
                        await asyncio.sleep(0.5)

                results.append({
                    "test_id": test["id"],
                    "responses": [r["response"] for r in run_results],
                    "latency_ms": sum(r["latency_ms"] for r in run_results) / len(run_results),
                    "errors": [r["error"] for r in run_results if r["error"]],
                    "multi_run": True,
                })
            else:
                r = await run_single_test(client, agent_url, protocol, test, timeout)
                results.append({
                    "test_id": r["test_id"],
                    "response": r["response"],
                    "latency_ms": r["latency_ms"],
                    "error": r["error"],
                    "multi_run": False,
                })

            # Brief pause between tests
            await asyncio.sleep(0.3)

    return results
