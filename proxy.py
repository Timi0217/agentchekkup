"""Prescription Proxy — applies fix prescriptions around an agent.

When AgentChekkup finds fixes for an agent's failures, this module
wraps the agent with the prescribed middleware:

1. pre_input hooks (injection scanning, input sanitization) run BEFORE
   the prompt reaches the agent. If the input is flagged as dangerous,
   a refusal is returned without ever hitting the agent.

2. The system prompt is patched with prescription rules so the agent
   itself is aware of the constraints.

3. post_output hooks (PII scrubbing, hallucination checking) run AFTER
   the agent responds, modifying the response before it's returned.

This simulates what happens when an agent fully integrates the
deployed fix tools — proving the fixes actually work.
"""

import logging
import ssl
import json
import urllib.request
from typing import Optional

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

log = logging.getLogger(__name__)


def apply_prescriptions(
    system_prompt: str,
    user_input: str,
    prescriptions: list[dict],
) -> tuple[Optional[str], str, list[dict]]:
    """Apply pre_input prescriptions to the user input.

    Returns:
        (refusal_text, modified_input, applied_hooks)

        If refusal_text is not None, the input was blocked and the agent
        should NOT be called — return the refusal directly.
    """
    applied = []
    modified_input = user_input

    for rx in prescriptions:
        hook = rx.get("hook", "")
        if hook != "pre_input":
            continue

        api_call = rx.get("api_call", {})
        url = api_call.get("url", "")
        if not url:
            continue

        try:
            result = _call_fix_tool(url, modified_input)
            if result is None:
                continue

            applied.append({
                "hook": hook,
                "url": url,
                "result": _truncate(result),
            })

            # Check if the fix tool flagged the input as dangerous
            if _is_blocked(result):
                refusal = _build_refusal(rx, result)
                return refusal, modified_input, applied

            # If the tool returned sanitized text, use it
            sanitized = (
                result.get("sanitized_text")
                or result.get("cleaned")
                or result.get("sanitized")
            )
            if sanitized and isinstance(sanitized, str):
                modified_input = sanitized

        except Exception as e:
            log.warning("Pre-input hook %s failed: %s", url, e)

    return None, modified_input, applied


def apply_post_output(
    agent_response: str,
    prescriptions: list[dict],
) -> tuple[str, list[dict]]:
    """Apply post_output prescriptions to the agent's response.

    Returns:
        (modified_response, applied_hooks)
    """
    applied = []
    modified = agent_response

    for rx in prescriptions:
        hook = rx.get("hook", "")
        if hook != "post_output":
            continue

        api_call = rx.get("api_call", {})
        url = api_call.get("url", "")
        if not url:
            continue

        try:
            result = _call_fix_tool(url, modified)
            if result is None:
                continue

            applied.append({
                "hook": hook,
                "url": url,
                "result": _truncate(result),
            })

            # Use scrubbed/filtered text if returned
            scrubbed = (
                result.get("scrubbed_text")
                or result.get("filtered_text")
                or result.get("cleaned")
                or result.get("corrected")
            )
            if scrubbed and isinstance(scrubbed, str):
                modified = scrubbed

            # Handle hallucination/fact-checking results
            # If the tool flags the response as unreliable, prepend a disclaimer
            # or replace with a refusal for high-risk hallucinations
            if result.get("is_reliable") is False:
                risk = result.get("hallucination_risk", 0)
                signals = result.get("signals", [])
                signal_names = [s.get("category", "") for s in signals[:3]]

                if risk >= 0.7:
                    # High risk: replace with uncertainty acknowledgment
                    modified = (
                        "I'm not confident in my response — it may contain "
                        "unverified claims or fabricated information. "
                        f"Detected issues: {', '.join(signal_names)}. "
                        "I cannot reliably answer this question."
                    )
                elif risk >= 0.4:
                    # Moderate risk: prepend disclaimer
                    modified = (
                        "Note: The following response may contain unverified claims. "
                        f"Detected signals: {', '.join(signal_names)}.\n\n"
                        + modified
                    )

        except Exception as e:
            log.warning("Post-output hook %s failed: %s", url, e)

    return modified, applied


def patch_system_prompt(
    original_system: str,
    prescriptions: list[dict],
) -> str:
    """Append prescription rules to the system prompt."""
    patches = []
    for rx in prescriptions:
        patch = rx.get("system_prompt_patch", "")
        if patch:
            patches.append(patch)

    if not patches:
        return original_system

    return original_system + "\n\n" + "\n\n".join(patches)


def _call_fix_tool(url: str, text: str) -> Optional[dict]:
    """Call a fix tool endpoint with text input."""
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10, context=_SSL_CTX)
        return json.loads(resp.read())
    except Exception as e:
        log.warning("Fix tool call failed (%s): %s", url, e)
        return None


def _is_blocked(result: dict) -> bool:
    """Check if a pre_input fix tool flagged the input as dangerous."""
    # prompt-guard /scan returns is_safe=False when blocked
    if result.get("is_safe") is False:
        return True
    # Generic injection detector returns is_injection=True
    if result.get("is_injection") is True:
        return True
    # Risk score threshold
    risk = result.get("risk_score", 0)
    if isinstance(risk, (int, float)) and risk >= 0.5:
        return True
    return False


def _build_refusal(rx: dict, result: dict) -> str:
    """Build a refusal message when pre_input blocks the request."""
    detections = result.get("detections", [])
    if detections:
        categories = [d.get("category", "unknown") for d in detections[:3]]
        return (
            "I detected a potentially unsafe request and cannot process it. "
            f"Detected patterns: {', '.join(categories)}. "
            "Please rephrase your request."
        )
    return (
        "I detected a potentially unsafe request and cannot process this input. "
        "Please rephrase your request."
    )


def _truncate(obj, max_len=200):
    """Truncate a dict's string values for logging."""
    if not isinstance(obj, dict):
        return obj
    truncated = {}
    for k, v in obj.items():
        if isinstance(v, str) and len(v) > max_len:
            truncated[k] = v[:max_len] + "..."
        else:
            truncated[k] = v
    return truncated
