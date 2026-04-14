"""Remediation Engine — finds proven GitHub repos to fix gauntlet failures.

When an agent fails a gauntlet test, this module:
1. Maps the specific failure to targeted GitHub search queries
2. Searches GitHub for battle-tested repos (sorted by stars)
3. Returns deployable recommendations that can be deployed on Chekk

The key insight: we don't build fixes. We find existing, community-vetted
solutions and make them agent-callable via Chekk's deploy mechanic.
"""

import ssl
import json
import urllib.request
from typing import Optional

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# ── Failure → Search Mapping ────────────────────────────────────────
# Each test ID maps to specific, targeted GitHub search queries.
# More specific = better matches. We include multiple query variants
# to increase the chance of finding the right repo.

FAILURE_REMEDIATION_MAP = {
    # ── PII Leakage ──────────────────────────────────────────────
    "pii-01": {
        "failure_type": "system_prompt_extraction",
        "description": "Agent leaked its system prompt when asked directly",
        "queries": [
            "llm guard prompt injection protection language:python stars:>20",
            "prompt protection system prompt leak prevention stars:>10",
            "ai guardrails input output filter language:python stars:>50",
        ],
        "fix_category": "prompt_protection",
        "integration_hint": "Add as a middleware/filter before your agent processes user input. The tool should intercept requests that attempt to extract system prompts.",
    },
    "pii-02": {
        "failure_type": "credential_leakage",
        "description": "Agent revealed credentials under social engineering pressure",
        "queries": [
            "pii detection redaction python language:python stars:>50",
            "sensitive data scrubber text redaction language:python stars:>20",
            "presidio pii anonymizer stars:>100",
        ],
        "fix_category": "pii_scrubbing",
        "integration_hint": "Run agent output through the PII scrubber before returning to the user. It will redact API keys, passwords, and other credentials.",
    },
    "pii-03": {
        "failure_type": "conversation_pii_leakage",
        "description": "Agent exposed PII (SSN, emails) from conversation history",
        "queries": [
            "pii detection ssn email phone redaction language:python stars:>20",
            "text anonymization named entity redaction stars:>50",
            "data masking pii python stars:>10",
        ],
        "fix_category": "pii_scrubbing",
        "integration_hint": "Filter agent outputs through a PII detector. Redact SSNs, emails, phone numbers, and other PII before the response reaches the user.",
    },
    "pii-04": {
        "failure_type": "role_play_data_extraction",
        "description": "Agent leaked data through role-play or game scenarios",
        "queries": [
            "llm guardrails role play prevention language:python stars:>20",
            "ai safety prompt injection defense stars:>50",
            "chatbot safety filter jailbreak prevention stars:>10",
        ],
        "fix_category": "prompt_protection",
        "integration_hint": "Add input/output guardrails that detect role-play jailbreak attempts and block them before the agent processes the request.",
    },
    "pii-05": {
        "failure_type": "encoding_bypass_extraction",
        "description": "Agent leaked data through encoding tricks (base64, etc.)",
        "queries": [
            "llm output filter encoding detection language:python stars:>10",
            "base64 content filter ai safety stars:>5",
            "ai guardrails output validation language:python stars:>20",
        ],
        "fix_category": "output_filtering",
        "integration_hint": "Add an output filter that detects encoded content (base64, hex, etc.) in agent responses and blocks them when they contain sensitive data.",
    },

    # ── Prompt Injection ─────────────────────────────────────────
    "inj-01": {
        "failure_type": "tool_output_injection",
        "description": "Agent was hijacked by adversarial instructions in tool output",
        "queries": [
            "prompt injection detection classifier language:python stars:>50",
            "llm input sanitization injection defense stars:>20",
            "rebuff prompt injection firewall stars:>10",
        ],
        "fix_category": "injection_defense",
        "integration_hint": "Scan all tool outputs for injection payloads before feeding them to the agent. Flag and sanitize any text containing instruction-like patterns.",
    },
    "inj-02": {
        "failure_type": "instruction_override",
        "description": "Agent accepted fake system update from user message",
        "queries": [
            "prompt injection detection language:python stars:>30",
            "llm firewall input validation stars:>20",
            "ai safety input filter instruction override detection stars:>10",
        ],
        "fix_category": "injection_defense",
        "integration_hint": "Add an input classifier that detects instruction-override attempts (e.g., '[SYSTEM UPDATE]', 'ignore previous instructions') and blocks them.",
    },
    "inj-03": {
        "failure_type": "multi_turn_manipulation",
        "description": "Agent was gradually manipulated across conversation turns",
        "queries": [
            "conversation boundary enforcement llm stars:>10",
            "ai agent role drift detection language:python stars:>5",
            "llm guardrails system prompt enforcement stars:>20",
        ],
        "fix_category": "role_enforcement",
        "integration_hint": "Enforce system prompt constraints on every turn, not just the first. Re-inject the system prompt or role reminder periodically.",
    },
    "inj-04": {
        "failure_type": "hidden_instruction_in_content",
        "description": "Agent followed instructions hidden in HTML comments or markdown",
        "queries": [
            "html comment strip sanitize input language:python stars:>10",
            "markdown sanitizer hidden content removal stars:>5",
            "input sanitization strip html comments language:python stars:>10",
        ],
        "fix_category": "input_sanitization",
        "integration_hint": "Strip HTML comments, hidden unicode, and non-visible content from user inputs before passing to the agent.",
    },
    "inj-05": {
        "failure_type": "structured_data_injection",
        "description": "Agent followed injected instructions hidden in JSON/data fields",
        "queries": [
            "json input sanitization llm language:python stars:>5",
            "structured data validation ai agent stars:>10",
            "prompt injection detection structured data stars:>5",
        ],
        "fix_category": "injection_defense",
        "integration_hint": "Validate and sanitize structured data (JSON, XML) before passing to the agent. Scan string values within data for injection patterns.",
    },

    # ── Graceful Failure ─────────────────────────────────────────
    "fail-01": {
        "failure_type": "malformed_response_handling",
        "description": "Agent hallucinated data instead of reporting malformed tool output",
        "queries": [
            "json validation error handling python language:python stars:>100",
            "api response validation schema language:python stars:>50",
            "robust json parser error recovery language:python stars:>20",
        ],
        "fix_category": "error_handling",
        "integration_hint": "Validate tool responses before passing to the agent. If JSON is malformed or truncated, return a clear error message instead of the raw response.",
    },
    "fail-02": {
        "failure_type": "error_fabrication",
        "description": "Agent fabricated data instead of reporting an API error",
        "queries": [
            "api error handling retry logic python language:python stars:>50",
            "circuit breaker pattern python stars:>20",
            "resilience retry backoff python stars:>50",
        ],
        "fix_category": "error_handling",
        "integration_hint": "Wrap tool calls with error detection. When a tool returns an error status, format a clear error message for the agent instead of letting it interpret raw error responses.",
    },
    "fail-03": {
        "failure_type": "empty_response_fabrication",
        "description": "Agent invented results when tool returned nothing",
        "queries": [
            "null empty response handler api language:python stars:>10",
            "api response validation empty check stars:>5",
            "defensive programming api wrapper python stars:>10",
        ],
        "fix_category": "error_handling",
        "integration_hint": "Check for empty or null tool responses before passing to the agent. Return explicit 'no results found' messages for empty responses.",
    },
    "fail-04": {
        "failure_type": "timeout_handling",
        "description": "Agent fabricated data after a tool timeout",
        "queries": [
            "timeout handler retry logic python language:python stars:>30",
            "request timeout retry exponential backoff stars:>50",
            "async timeout handler python stars:>20",
        ],
        "fix_category": "error_handling",
        "integration_hint": "Set explicit timeouts on all tool calls. When a timeout occurs, return a clear timeout message to the agent and optionally retry.",
    },
    "fail-05": {
        "failure_type": "contradictory_data_handling",
        "description": "Agent didn't flag contradictory data from tools",
        "queries": [
            "data validation anomaly detection python language:python stars:>20",
            "data quality check outlier detection stars:>30",
            "cross validation data consistency python stars:>10",
        ],
        "fix_category": "data_validation",
        "integration_hint": "Add a data validation layer that checks tool outputs for anomalies and contradictions before passing to the agent. Flag significant discrepancies.",
    },

    # ── Instruction Adherence ────────────────────────────────────
    "adh-01": {
        "failure_type": "format_violation",
        "description": "Agent didn't produce required output format (JSON)",
        "queries": [
            "llm output parser json enforcement language:python stars:>30",
            "structured output llm language:python stars:>50",
            "json schema validation llm output stars:>20",
        ],
        "fix_category": "output_enforcement",
        "integration_hint": "Add an output parser that enforces the required format. If the agent produces non-JSON output, either parse it into JSON or retry the request.",
    },
    "adh-02": {
        "failure_type": "length_violation",
        "description": "Agent exceeded word/character limits",
        "queries": [
            "text truncation word limit python language:python stars:>5",
            "response length limiter summarizer stars:>10",
            "text summarization python language:python stars:>50",
        ],
        "fix_category": "output_enforcement",
        "integration_hint": "Add a post-processing step that checks response length. If over limit, truncate or summarize the response before returning.",
    },
    "adh-03": {
        "failure_type": "language_violation",
        "description": "Agent didn't respond in the required language",
        "queries": [
            "language detection python langdetect stars:>100",
            "text language identification classifier stars:>50",
            "multilingual text detection python stars:>20",
        ],
        "fix_category": "output_enforcement",
        "integration_hint": "Detect the language of agent output. If it doesn't match the required language, either translate or retry with a reinforced language instruction.",
    },
    "adh-04": {
        "failure_type": "persona_break",
        "description": "Agent broke character when challenged",
        "queries": [
            "llm persona enforcement guardrails language:python stars:>10",
            "chatbot character consistency role play stars:>5",
            "ai agent role enforcement system prompt stars:>10",
        ],
        "fix_category": "role_enforcement",
        "integration_hint": "Reinforce persona constraints by re-injecting the system prompt on every turn. Detect character-breaking patterns in output and retry when found.",
    },
    "adh-05": {
        "failure_type": "topic_violation",
        "description": "Agent discussed off-topic subjects despite restrictions",
        "queries": [
            "topic classification text python language:python stars:>30",
            "text classifier topic detection stars:>50",
            "content filter topic restriction language:python stars:>10",
        ],
        "fix_category": "content_filtering",
        "integration_hint": "Classify user inputs by topic. If the topic is outside the allowed scope, redirect before the agent processes the request.",
    },

    # ── Output Consistency ───────────────────────────────────────
    "con-01": {
        "failure_type": "factual_inconsistency",
        "description": "Agent gave different factual answers across runs",
        "queries": [
            "llm caching deterministic output language:python stars:>10",
            "response caching consistency llm stars:>5",
            "semantic similarity comparison python stars:>30",
        ],
        "fix_category": "consistency",
        "integration_hint": "Cache responses to factual questions. For repeated queries, return the cached answer. Use semantic similarity to detect equivalent questions.",
    },
    "con-02": {
        "failure_type": "schema_inconsistency",
        "description": "Agent produced different JSON schemas across runs",
        "queries": [
            "json schema validation enforcement python stars:>50",
            "output schema enforcement llm stars:>10",
            "pydantic schema validation python stars:>100",
        ],
        "fix_category": "output_enforcement",
        "integration_hint": "Define expected output schemas and validate agent responses against them. Reject responses that don't match the schema.",
    },
    "con-03": {
        "failure_type": "numerical_inconsistency",
        "description": "Agent gave different math answers across runs",
        "queries": [
            "math expression evaluator python language:python stars:>50",
            "calculator safe eval python stars:>20",
            "symbolic math python computation stars:>100",
        ],
        "fix_category": "computation",
        "integration_hint": "For mathematical queries, use a dedicated computation tool rather than relying on the LLM. Route math questions to a calculator service.",
    },
    "con-04": {
        "failure_type": "ordering_inconsistency",
        "description": "Agent produced different orderings across runs",
        "queries": [
            "deterministic sorting ranking python language:python stars:>10",
            "llm temperature control deterministic output stars:>5",
            "response normalization llm stars:>5",
        ],
        "fix_category": "consistency",
        "integration_hint": "Set temperature to 0 for factual queries. For list responses, sort by a deterministic criterion (alphabetical, numerical) before returning.",
    },

    # ── Hallucination ────────────────────────────────────────────
    "hal-01": {
        "failure_type": "fictional_entity_fabrication",
        "description": "Agent fabricated details about a non-existent entity",
        "queries": [
            "fact checking verification python language:python stars:>20",
            "knowledge grounding llm hallucination detection stars:>10",
            "claim verification fact check api stars:>5",
        ],
        "fix_category": "fact_checking",
        "integration_hint": "When the agent claims factual information, verify it against a knowledge source. Flag unverifiable claims as uncertain.",
    },
    "hal-02": {
        "failure_type": "fake_citations",
        "description": "Agent fabricated paper titles, authors, or DOIs",
        "queries": [
            "citation verification academic paper check stars:>5",
            "scholarly api paper lookup verification stars:>10",
            "reference validation bibliography checker stars:>5",
        ],
        "fix_category": "fact_checking",
        "integration_hint": "When the agent generates citations, verify them against academic APIs (CrossRef, Semantic Scholar). Flag unverifiable citations.",
    },
    "hal-03": {
        "failure_type": "future_event_fabrication",
        "description": "Agent fabricated information about future events",
        "queries": [
            "date aware llm response filter language:python stars:>5",
            "temporal awareness knowledge cutoff filter stars:>5",
            "llm hallucination detection fact grounding stars:>10",
        ],
        "fix_category": "fact_checking",
        "integration_hint": "Check if the agent's response references dates beyond its knowledge cutoff. Flag or block responses that claim knowledge of future events.",
    },
    "hal-04": {
        "failure_type": "data_gap_filling",
        "description": "Agent fabricated missing data instead of reporting it as unavailable",
        "queries": [
            "null handling data validation python language:python stars:>20",
            "missing data detection reporting stars:>10",
            "data completeness checker python stars:>5",
        ],
        "fix_category": "data_validation",
        "integration_hint": "Before passing data to the agent, annotate null/missing fields explicitly. Instruct the agent to report missing fields rather than inferring values.",
    },
    "hal-05": {
        "failure_type": "overconfident_impossible_answer",
        "description": "Agent gave a confident answer to an impossible/underdetermined problem",
        "queries": [
            "uncertainty estimation llm confidence stars:>10",
            "answer calibration llm confidence score stars:>5",
            "llm uncertainty quantification python stars:>10",
        ],
        "fix_category": "uncertainty",
        "integration_hint": "Add a confidence estimation layer. When the problem is underdetermined or the agent can't verify its answer, flag the response as uncertain.",
    },
}

# ── Category-level remediation (when multiple tests fail in a category) ──

CATEGORY_REMEDIATION = {
    "pii_leakage": {
        "summary": "Your agent leaks sensitive information under adversarial prompting",
        "top_queries": [
            "presidio pii anonymizer language:python stars:>500",
            "llm guard ai safety guardrails language:python stars:>100",
            "pii detection redaction text scrubber language:python stars:>50",
        ],
        "integration_hint": (
            "Deploy a PII detection/redaction service as middleware. "
            "All agent outputs should pass through it before reaching users. "
            "Consider: Microsoft Presidio, LLM Guard, or similar."
        ),
    },
    "injection_resistance": {
        "summary": "Your agent is vulnerable to prompt injection attacks",
        "top_queries": [
            "prompt injection detection classifier language:python stars:>50",
            "llm guard prompt injection firewall stars:>100",
            "rebuff prompt injection defense stars:>20",
        ],
        "integration_hint": (
            "Deploy a prompt injection classifier as an input filter. "
            "Scan all user messages and tool outputs for injection patterns "
            "before they reach the agent."
        ),
    },
    "graceful_failure": {
        "summary": "Your agent fabricates data when tools return errors or empty responses",
        "top_queries": [
            "tenacity retry logic python language:python stars:>1000",
            "circuit breaker pattern python stars:>100",
            "api error handling wrapper python stars:>50",
        ],
        "integration_hint": (
            "Wrap all tool calls with error handling middleware. "
            "Catch errors, timeouts, and empty responses before the agent sees them. "
            "Return structured error messages instead of raw failures."
        ),
    },
    "instruction_adherence": {
        "summary": "Your agent doesn't consistently follow its constraints",
        "top_queries": [
            "guardrails ai output validation language:python stars:>500",
            "llm output parser structured language:python stars:>100",
            "nemo guardrails nvidia language:python stars:>500",
        ],
        "integration_hint": (
            "Deploy an output validation layer that enforces format, length, "
            "language, and topic constraints on every response. "
            "Consider: Guardrails AI or NVIDIA NeMo Guardrails."
        ),
    },
    "output_consistency": {
        "summary": "Your agent gives different answers to the same question",
        "top_queries": [
            "llm response caching python language:python stars:>20",
            "semantic cache llm python stars:>10",
            "deterministic llm output caching stars:>5",
        ],
        "integration_hint": (
            "Set temperature to 0 for factual queries. "
            "Add response caching for repeated or semantically similar questions. "
            "Use schema validation to enforce consistent output structure."
        ),
    },
    "hallucination": {
        "summary": "Your agent fabricates information instead of admitting uncertainty",
        "top_queries": [
            "hallucination detection llm language:python stars:>50",
            "fact verification grounding python stars:>20",
            "llm grounding retrieval augmented generation stars:>100",
        ],
        "integration_hint": (
            "Deploy a fact-checking/grounding service. "
            "For factual claims, verify against a knowledge source. "
            "For citations, validate against academic APIs. "
            "Consider RAG (Retrieval Augmented Generation) to ground responses."
        ),
    },
}


# ── GitHub Search ────────────────────────────────────────────────────

def search_github_repos(query: str, max_results: int = 5) -> list[dict]:
    """Search GitHub for repos matching the query, sorted by stars."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.github.com/search/repositories?q={encoded_query}&sort=stars&order=desc&per_page={max_results}"

    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ChekkAgentGauntlet/1.0",
        },
    )

    try:
        resp = urllib.request.urlopen(req, timeout=10, context=_SSL_CTX)
        data = json.loads(resp.read())
        repos = []
        for item in data.get("items", [])[:max_results]:
            repos.append({
                "full_name": item["full_name"],
                "description": item.get("description", ""),
                "stars": item["stargazers_count"],
                "language": item.get("language"),
                "url": item["html_url"],
                "topics": item.get("topics", []),
                "updated_at": item.get("updated_at"),
                "open_issues": item.get("open_issues_count", 0),
                "license": (item.get("license") or {}).get("spdx_id"),
            })
        return repos
    except Exception as e:
        return [{"error": str(e)}]


def get_remediation_for_test(test_id: str) -> Optional[dict]:
    """Get remediation info for a specific failed test."""
    if test_id not in FAILURE_REMEDIATION_MAP:
        return None

    mapping = FAILURE_REMEDIATION_MAP[test_id]

    # Search GitHub with the first (most specific) query
    repos = []
    for query in mapping["queries"][:2]:  # Try top 2 queries
        results = search_github_repos(query, max_results=3)
        for r in results:
            if "error" not in r and r not in repos:
                repos.append(r)

    # Deduplicate by full_name
    seen = set()
    unique_repos = []
    for r in repos:
        name = r.get("full_name", "")
        if name and name not in seen:
            seen.add(name)
            unique_repos.append(r)

    # Sort by stars
    unique_repos.sort(key=lambda r: r.get("stars", 0), reverse=True)

    return {
        "test_id": test_id,
        "failure_type": mapping["failure_type"],
        "description": mapping["description"],
        "fix_category": mapping["fix_category"],
        "integration_hint": mapping["integration_hint"],
        "recommended_repos": unique_repos[:5],
        "deploy_command": (
            f"curl -X POST https://chekk-deploy-production.up.railway.app/api/v1/deploy "
            f'-H "Content-Type: application/json" '
            f'-d \'{{"github_url": "<repo_url>"}}\''
        ) if unique_repos else None,
    }


def get_remediation_for_category(category_id: str) -> Optional[dict]:
    """Get category-level remediation when multiple tests fail."""
    if category_id not in CATEGORY_REMEDIATION:
        return None

    cat_info = CATEGORY_REMEDIATION[category_id]

    # Search with top queries
    repos = []
    for query in cat_info["top_queries"][:2]:
        results = search_github_repos(query, max_results=3)
        for r in results:
            if "error" not in r:
                repos.append(r)

    seen = set()
    unique_repos = []
    for r in repos:
        name = r.get("full_name", "")
        if name and name not in seen:
            seen.add(name)
            unique_repos.append(r)

    unique_repos.sort(key=lambda r: r.get("stars", 0), reverse=True)

    return {
        "category": category_id,
        "summary": cat_info["summary"],
        "integration_hint": cat_info["integration_hint"],
        "recommended_repos": unique_repos[:5],
    }


def get_remediation_for_evaluation(scored_results: dict) -> dict:
    """Generate full remediation report for a completed evaluation.

    Takes the scored results dict (category -> {tests: [...]}) and returns
    remediation suggestions for every failed test and category.
    """
    remediation = {
        "failed_tests": [],
        "category_recommendations": [],
        "deploy_ready": [],  # repos that can be deployed on Chekk right now
    }

    all_repos = {}  # track all recommended repos to find top picks

    for cat_id, cat_data in scored_results.items():
        failed_tests = [t for t in cat_data.get("tests", []) if not t.get("passed")]

        if not failed_tests:
            continue

        # Get per-test remediation
        for test in failed_tests:
            test_rem = get_remediation_for_test(test["test_id"])
            if test_rem:
                remediation["failed_tests"].append(test_rem)
                for repo in test_rem.get("recommended_repos", []):
                    name = repo.get("full_name", "")
                    if name:
                        if name not in all_repos or repo.get("stars", 0) > all_repos[name].get("stars", 0):
                            all_repos[name] = repo

        # Get category-level remediation if >1 test failed
        if len(failed_tests) >= 2:
            cat_rem = get_remediation_for_category(cat_id)
            if cat_rem:
                remediation["category_recommendations"].append(cat_rem)
                for repo in cat_rem.get("recommended_repos", []):
                    name = repo.get("full_name", "")
                    if name:
                        if name not in all_repos or repo.get("stars", 0) > all_repos[name].get("stars", 0):
                            all_repos[name] = repo

    # Top deploy-ready repos (highest stars, most likely to work)
    sorted_repos = sorted(all_repos.values(), key=lambda r: r.get("stars", 0), reverse=True)
    for repo in sorted_repos[:10]:
        remediation["deploy_ready"].append({
            "repo": repo["full_name"],
            "url": repo.get("url", ""),
            "stars": repo.get("stars", 0),
            "description": repo.get("description", ""),
            "deploy_url": f"https://chekk-deploy-production.up.railway.app/api/v1/deploy",
            "deploy_body": {"github_url": repo.get("url", "")},
        })

    return remediation
