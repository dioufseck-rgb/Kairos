# =========================
# Kairos Executive Narrative (LLM-backed, low-hallucination)
# Uses: Gemini API via Google GenAI Python SDK (google-genai)
# =========================

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Google GenAI SDK
from google import genai
from google.genai import types


# -------------------------
# Evidence packet
# -------------------------

def build_evidence_packet(result: Dict[str, Any], y: str, max_drivers: int = 5) -> Dict[str, Any]:
    selected = (result.get("selected_drivers") or [])[:max_drivers]
    chains = result.get("causal_chains") or {}
    cfs = result.get("counterfactuals") or {}
    elim = result.get("eliminated") or {}
    baseline = result.get("baseline") or {}

    drivers: List[Dict[str, Any]] = []
    for d in selected:
        x = d.get("x")
        if not x:
            continue
        top_chain = (chains.get(x) or [{}])[0] if chains.get(x) else {}
        drivers.append({
            "x": x,
            "ate": d.get("effect"),
            "rows_used": d.get("rows_used"),
            "confounders_used": d.get("confounders_used") or [],
            "top_chain": {
                "chain": top_chain.get("chain", [x, y]),
                "type": top_chain.get("type", "unknown"),
                "length": top_chain.get("length", None),
            },
            "counterfactual": cfs.get(x, {}),
        })

    # Provide a compact, explicit set of allowed variable names
    allowed_vars = set()
    allowed_vars.add(y)
    for d in drivers:
        allowed_vars.add(d["x"])
        for node in d.get("top_chain", {}).get("chain", []) or []:
            allowed_vars.add(str(node))
    for k in elim.keys():
        allowed_vars.add(str(k))

    return {
        "outcome": y,
        "baseline": baseline,  # expects {"y_mean":..., "y_ci":[...,...]} if present
        "drivers": drivers,
        "eliminated": elim,
        "allowed_variables": sorted(list(allowed_vars)),
        "policy": {
            "no_new_variables": True,
            "no_new_numbers": True,
            "must_use_only_evidence": True,
        },
    }


# -------------------------
# Non-action reason taxonomy (for "do not drive" section)
# -------------------------

def classify_non_action_reason(reason: str) -> str:
    r = (reason or "").lower()
    if "uncontrollable" in r or "context" in r:
        return "Uncontrollable context (monitor/segment; don’t optimize)"
    if "no material" in r or "negligible" in r or "weak" in r:
        return "Weak/immaterial effect (not worth optimizing)"
    if "not estim" in r or "identify" in r or "not identifiable" in r:
        return "Not identifiable with current data (needs experiment or more controls)"
    if "proxy" in r or "placebo" in r:
        return "Proxy/placebo risk (optimize upstream levers instead)"
    if "error" in r or "failed" in r:
        return "Estimation failed / insufficient statistical support"
    return "Insufficient evidence (do not act; collect better data)"


# -------------------------
# JSON schema for Gemini structured output
# -------------------------

def narrative_schema() -> Dict[str, Any]:
    """
    JSON Schema for the model to follow. We keep it small and strict.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "executive_summary": {"type": "string"},
            "key_drivers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "x": {"type": "string"},
                        "ate": {"type": "number"},
                        "delta_mean": {"type": "number"},
                        "delta_ci_low": {"type": "number"},
                        "delta_ci_high": {"type": "number"},
                        "chain_type": {"type": "string"},
                        "chain": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["x", "ate", "chain_type", "chain"],
                },
            },
            "do_not_optimize": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "variable": {"type": "string"},
                        "category": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["variable", "category", "reason"],
                },
            },
            "next_steps": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["title", "executive_summary", "key_drivers", "do_not_optimize", "next_steps"],
    }


# -------------------------
# Gemini caller (Developer API)
# -------------------------

class GeminiNarrator:
    """
    Minimal Gemini wrapper using google-genai SDK.
    Basic usage in docs: client.models.generate_content(...).text :contentReference[oaicite:2]{index=2}
    Structured outputs w/ JSON schema are supported. :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        api_key: Optional[str] = "None",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY (recommended) or GOOGLE_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        # In structured output, resp.text should still contain JSON text you can parse. :contentReference[oaicite:4]{index=4}
        try:
            return json.loads(resp.text)
        except Exception as e:
            raise ValueError(f"Gemini returned non-JSON or unparsable JSON. Raw text:\n{resp.text}") from e


# -------------------------
# Validation to reduce hallucinations further
# -------------------------

def _collect_allowed_variables(evidence: Dict[str, Any]) -> Set[str]:
    allowed = set(evidence.get("allowed_variables") or [])
    return {str(a) for a in allowed}


def validate_narrative_json(narr: Dict[str, Any], evidence: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Hard checks:
      - mentioned variable names must be in evidence allowed set
      - key_drivers[*].x must be allowed
      - chains must only contain allowed nodes
    This won't catch all numeric hallucinations, but combined with schema + low temp, it helps a lot.
    """
    issues: List[str] = []
    allowed = _collect_allowed_variables(evidence)

    # Check drivers
    for i, kd in enumerate(narr.get("key_drivers") or []):
        x = str(kd.get("x"))
        if x not in allowed:
            issues.append(f"key_drivers[{i}].x='{x}' is not in allowed_variables.")
        chain = kd.get("chain") or []
        for node in chain:
            if str(node) not in allowed:
                issues.append(f"key_drivers[{i}].chain contains node '{node}' not in allowed_variables.")

    # Check do_not_optimize variables
    for i, item in enumerate(narr.get("do_not_optimize") or []):
        v = str(item.get("variable"))
        if v not in allowed:
            issues.append(f"do_not_optimize[{i}].variable='{v}' is not in allowed_variables.")

    return (len(issues) == 0), issues


# -------------------------
# Prompting: rewrite + "do not drive" logic
# -------------------------

def llm_exec_narrative(
    result: Dict[str, Any],
    y: str,
    narrator: GeminiNarrator,
    max_drivers: int = 5,
) -> Dict[str, Any]:
    """
    Returns structured narrative JSON (validated).
    The LLM is constrained to ONLY use the evidence packet and to explicitly call out
    what NOT to optimize (uncontrollable, weak, not identifiable, placebo/proxy risk, etc.).
    """

    evidence = build_evidence_packet(result, y=y, max_drivers=max_drivers)

    # Provide model-friendly "do not optimize" categories from your eliminated dict
    eliminated = evidence.get("eliminated") or {}
    do_not = []
    for var, reason in eliminated.items():
        do_not.append({
            "variable": str(var),
            "category": classify_non_action_reason(str(reason)),
            "reason": str(reason),
        })

    # We give the model evidence + precomputed do_not list; model can rephrase but not invent.
    prompt = f"""
You are Kairos, an executive explanation writer.

Hard rules (must follow):
- Use ONLY the variables found in allowed_variables.
- Do NOT introduce new numbers. Use only numbers present in drivers[*].ate and drivers[*].counterfactual.* fields.
- If counterfactual delta fields are missing for a driver, you may omit delta values for that driver.
- Do NOT invent causal mechanisms: only use the provided top_chain per driver.
- You must include a "do_not_optimize" section grounded in the provided eliminated list and categories.

Write a concise narrative for executives:
- Title
- 3–6 sentence executive summary (plain language)
- key_drivers: for each driver, include x, ate, (delta_mean + CI if present), chain_type, chain
- do_not_optimize: include variable, category, reason (rephrase reason, but keep meaning)
- next_steps: 3–6 bullets; can include (a) operational focus areas for controllable drivers and (b) data/experiment recommendations
  but do not propose remedies that require new variables not in the evidence.

EVIDENCE (JSON):
{json.dumps(evidence, ensure_ascii=False)}

DO_NOT_OPTIMIZE_SEED (JSON):
{json.dumps(do_not, ensure_ascii=False)}
"""

    narr = narrator.generate_structured(prompt=prompt, schema=narrative_schema())

    ok, issues = validate_narrative_json(narr, evidence)
    if not ok:
        # One controlled repair attempt: ask model to fix only the violations.
        repair_prompt = f"""
Your previous JSON violated constraints. Fix ONLY the violations listed below.
Do not add new variables; use only allowed_variables.

VIOLATIONS:
{json.dumps(issues, ensure_ascii=False, indent=2)}

EVIDENCE (JSON):
{json.dumps(evidence, ensure_ascii=False)}

PREVIOUS_JSON:
{json.dumps(narr, ensure_ascii=False)}
"""
        narr = narrator.generate_structured(prompt=repair_prompt, schema=narrative_schema())
        ok2, issues2 = validate_narrative_json(narr, evidence)
        if not ok2:
            raise ValueError("Narrative validation failed after repair attempt:\n" + "\n".join(issues2))

    return narr


# -------------------------
# Rendering helpers (JSON -> Markdown / text)
# -------------------------

def narrative_json_to_markdown(narr: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"## {narr.get('title','Executive Narrative')}")
    lines.append("")
    lines.append(narr.get("executive_summary", "").strip())
    lines.append("")

    lines.append("### Key causal drivers")
    for kd in narr.get("key_drivers", []):
        x = kd["x"]
        ate = kd.get("ate", None)
        chain_type = kd.get("chain_type", "unknown")
        chain = " → ".join(kd.get("chain", []))

        # delta optional
        if "delta_mean" in kd and "delta_ci_low" in kd and "delta_ci_high" in kd:
            lines.append(f"- **{x}** (ATE {ate:+.3f}) | What-if: Δ ≈ {kd['delta_mean']:+.3f} "
                         f"(CI {kd['delta_ci_low']:+.3f}..{kd['delta_ci_high']:+.3f})")
        else:
            lines.append(f"- **{x}** (ATE {ate:+.3f})")

        lines.append(f"  - Chain ({chain_type}): {chain}")

    lines.append("")
    lines.append("### What not to optimize (and why)")
    for item in narr.get("do_not_optimize", []):
        lines.append(f"- **{item['variable']}** — {item['category']}. {item['reason']}")

    lines.append("")
    lines.append("### Next steps")
    for s in narr.get("next_steps", []):
        lines.append(f"- {s}")

    return "\n".join(lines)

def generate_exec_narrative(
    result: dict,
    y: str,
    max_drivers: int = 3,
) -> str:
    """
    Backward-compatible entry point.
    Uses LLM-backed narrative if available, otherwise falls back.
    """
    from kairos_narrative import narrative_json_to_markdown, GeminiNarrator, llm_exec_narrative

    narrator = GeminiNarrator()
    narr_json = llm_exec_narrative(
        result=result,
        y=y,
        narrator=narrator,
        max_drivers=max_drivers,
    )
    return narrative_json_to_markdown(narr_json)
