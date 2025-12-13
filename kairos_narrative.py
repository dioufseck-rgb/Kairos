# =========================
# Kairos Executive Narrative (LLM-backed, low-hallucination)
# Gemini API via Google GenAI Python SDK (google-genai)
# =========================

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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
                "length": top_chain.get("length"),
            },
            "counterfactual": cfs.get(x, {}),
        })

    allowed_vars: Set[str] = set()
    allowed_vars.add(str(y))
    for d in drivers:
        allowed_vars.add(str(d["x"]))
        for node in d.get("top_chain", {}).get("chain", []) or []:
            allowed_vars.add(str(node))
    for k in elim.keys():
        allowed_vars.add(str(k))

    return {
        "outcome": y,
        "baseline": baseline,  # if available: {"y_mean":..., "y_ci":[low,high]}
        "drivers": drivers,
        "eliminated": elim,
        "allowed_variables": sorted(list(allowed_vars)),
        "policy": {
            "no_new_variables": True,
            "no_new_numbers": True,
            "must_use_only_evidence": True,
        },
    }


def compact_evidence_for_llm(evidence: Dict[str, Any], *, max_eliminated: int = 10) -> Dict[str, Any]:
    """
    Reduce prompt size to avoid model truncation.
    Keep only what the narrative needs.
    """
    e = dict(evidence)

    # Remove very long lists; we validate post hoc anyway.
    e.pop("allowed_variables", None)
    e.pop("policy", None)

    elim = e.get("eliminated") or {}
    if isinstance(elim, dict) and len(elim) > max_eliminated:
        keys = list(elim.keys())[:max_eliminated]  # deterministic truncation
        e["eliminated"] = {k: elim[k] for k in keys}

    # Trim counterfactual payloads
    for d in (e.get("drivers") or []):
        cf = d.get("counterfactual")
        if isinstance(cf, dict) and cf:
            keep = {}
            for k in ("delta_mean", "delta_ci", "counterfactual_mean", "counterfactual_ci", "intervention"):
                if k in cf:
                    keep[k] = cf[k]
            d["counterfactual"] = keep

    return e


# -------------------------
# Non-action reason taxonomy
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
# Gemini structured output schema (NO additionalProperties!)
# -------------------------

def narrative_schema() -> Dict[str, Any]:
    """
    Keep schema simple: Gemini response_schema rejects some JSON Schema keywords
    (e.g., additionalProperties). Enforce strictness via our own validators.
    """
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "executive_summary": {"type": "string"},
            "key_drivers": {
                "type": "array",
                "items": {
                    "type": "object",
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
                    "properties": {
                        "variable": {"type": "string"},
                        "category": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["variable", "category", "reason"],
                },
            },
            "next_steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "executive_summary", "key_drivers", "do_not_optimize", "next_steps"],
    }


# -------------------------
# Gemini caller (robust JSON handling)
# -------------------------

class GeminiNarrator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,        # deterministic JSON
        max_output_tokens: int = 2500,   # reduce truncation
    ):
        self.api_key = api_key or os.getenv("GEMINI_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key or not str(self.api_key).strip():
            raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY (recommended) or GOOGLE_API_KEY.")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """
        Extract the largest {...} JSON object from a string.
        If truncated, returns from first '{' to end as fallback.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]  # truncated object

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            pass

        cand = GeminiNarrator._extract_json_object(text)
        if cand:
            try:
                return json.loads(cand)
            except Exception:
                return None
        return None

    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        def call(contents: str, temp: float) -> str:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=self.max_output_tokens,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            raw = (getattr(resp, "text", "") or "").strip()
            raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
            return raw

        # Attempt 1
        raw1 = call(prompt, self.temperature)
        obj = self._try_parse_json(raw1)
        if obj is not None:
            return obj

        # Attempt 2: repair (rewrite JSON cleanly)
        repair_prompt = f"""
Return ONLY valid JSON matching the schema. No markdown. No commentary.
Your previous response had invalid or truncated JSON.

PREVIOUS_RESPONSE:
{raw1}
"""
        raw2 = call(repair_prompt, 0.0)
        obj = self._try_parse_json(raw2)
        if obj is not None:
            return obj

        # Attempt 3: salvage — complete the JSON instead of rewriting
        candidate = self._extract_json_object(raw2) or self._extract_json_object(raw1) or raw2 or raw1
        complete_prompt = f"""
You will be given a PARTIAL JSON object that may be truncated.
Task: OUTPUT a COMPLETE, VALID JSON object matching the schema.
Do NOT add any new variables or numbers. Preserve keys/values; only fix truncation/escaping.

PARTIAL_JSON:
{candidate}
"""
        raw3 = call(complete_prompt, 0.0)
        obj = self._try_parse_json(raw3)
        if obj is not None:
            return obj

        raise ValueError(
            "Gemini returned non-JSON after 3 attempts.\n"
            f"RAW1:\n{raw1[:2000]}\n\nRAW2:\n{raw2[:2000]}\n\nRAW3:\n{raw3[:2000]}"
        )


# -------------------------
# Validation + Guardrails (anti-hallucination)
# -------------------------

def _collect_allowed_variables(evidence: Dict[str, Any]) -> Set[str]:
    return {str(a) for a in (evidence.get("allowed_variables") or [])}


def validate_narrative_json(narr: Dict[str, Any], evidence: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    allowed = _collect_allowed_variables(evidence)

    for i, kd in enumerate(narr.get("key_drivers") or []):
        x = str(kd.get("x"))
        if x not in allowed:
            issues.append(f"key_drivers[{i}].x='{x}' not in allowed_variables.")
        for node in (kd.get("chain") or []):
            if str(node) not in allowed:
                issues.append(f"key_drivers[{i}].chain node '{node}' not in allowed_variables.")

    for i, item in enumerate(narr.get("do_not_optimize") or []):
        v = str(item.get("variable"))
        if v not in allowed:
            issues.append(f"do_not_optimize[{i}].variable='{v}' not in allowed_variables.")

    return (len(issues) == 0), issues


def sanitize_do_not_optimize(
    narr: Dict[str, Any],
    allowed: Set[str],
    do_not_seed: List[Dict[str, str]],
    *,
    max_items: int = 8
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Remove invented do_not_optimize variables and replace with evidence-seeded ones.
    Returns (narr, fixes_applied).
    """
    fixes: List[str] = []
    cleaned: List[Dict[str, Any]] = []

    for item in (narr.get("do_not_optimize") or []):
        v = str(item.get("variable", "")).strip()
        if v in allowed:
            cleaned.append(item)
        else:
            fixes.append(f"Removed invented do_not_optimize variable: '{v}'")

    if not cleaned:
        cleaned = [d for d in do_not_seed if d["variable"] in allowed][:max_items]
        fixes.append("Replaced do_not_optimize with evidence-seeded list (model invented variables).")
    else:
        cleaned = cleaned[:max_items]
        existing = {str(x.get("variable")) for x in cleaned}
        for d in do_not_seed:
            if d["variable"] in allowed and d["variable"] not in existing and len(cleaned) < max_items:
                cleaned.append(d)
                existing.add(d["variable"])

    narr["do_not_optimize"] = cleaned
    return narr, fixes


# -------------------------
# Main LLM narrative builder
# -------------------------

def llm_exec_narrative(
    result: Dict[str, Any],
    y: str,
    narrator: GeminiNarrator,
    max_drivers: int = 5,
) -> Dict[str, Any]:
    evidence = build_evidence_packet(result, y=y, max_drivers=max_drivers)
    allowed = _collect_allowed_variables(evidence)

    eliminated = evidence.get("eliminated") or {}
    do_not_seed = [{
        "variable": str(var),
        "category": classify_non_action_reason(str(reason)),
        "reason": str(reason),
    } for var, reason in eliminated.items()]

    do_not_vars = [d["variable"] for d in do_not_seed]
    evidence_compact = compact_evidence_for_llm(evidence, max_eliminated=10)

    prompt = f"""
You are Kairos, an executive explanation writer.

HARD RULES:
- Output MUST be a single valid JSON object (no markdown, no commentary).
- Use ONLY variables from the evidence. Do NOT invent variable names.
- For do_not_optimize.variable, you MUST choose ONLY from DO_NOT_VARIABLES (closed set).
- Do NOT introduce new numbers (only reuse numbers present in evidence).
- Do NOT invent causal mechanisms; use only top_chain provided for each driver.
- Keep it concise.

FORMAT LIMITS:
- executive_summary: <= 90 words
- key_drivers: <= {max_drivers} items
- next_steps: 3 to 5 bullets, each <= 18 words
- do_not_optimize: <= 8 items, each reason <= 22 words

DO_NOT_VARIABLES (closed set):
{json.dumps(do_not_vars, ensure_ascii=False)}

EVIDENCE (JSON):
{json.dumps(evidence_compact, ensure_ascii=False)}
"""

    narr = narrator.generate_structured(prompt=prompt, schema=narrative_schema())
    narr, _ = sanitize_do_not_optimize(narr, allowed, do_not_seed)

    ok, issues = validate_narrative_json(narr, evidence)
    if not ok:
        repair_prompt = f"""
Fix ONLY these violations. Do not add any new variables.
For do_not_optimize, you MUST choose variables ONLY from DO_NOT_VARIABLES.
Return ONLY a single JSON object. No markdown.

DO_NOT_VARIABLES:
{json.dumps(do_not_vars, ensure_ascii=False)}

VIOLATIONS:
{json.dumps(issues, ensure_ascii=False, indent=2)}

EVIDENCE (JSON):
{json.dumps(evidence_compact, ensure_ascii=False)}

PREVIOUS_JSON:
{json.dumps(narr, ensure_ascii=False)}
"""
        narr = narrator.generate_structured(prompt=repair_prompt, schema=narrative_schema())
        narr, _ = sanitize_do_not_optimize(narr, allowed, do_not_seed)

        ok2, issues2 = validate_narrative_json(narr, evidence)
        if not ok2:
            # deterministic fallback: replace do_not_optimize with seed
            narr["do_not_optimize"] = [d for d in do_not_seed if d["variable"] in allowed][:8]
            ok3, issues3 = validate_narrative_json(narr, evidence)
            if not ok3:
                raise ValueError("Narrative validation failed after repair:\n" + "\n".join(issues3))

    return narr


def narrative_json_to_markdown(narr: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"## {narr.get('title', 'Executive Narrative')}")
    lines.append("")
    lines.append((narr.get("executive_summary") or "").strip())
    lines.append("")

    lines.append("### Key causal drivers")
    for kd in (narr.get("key_drivers") or []):
        x = kd.get("x", "")
        ate = float(kd.get("ate", 0.0))
        chain_type = kd.get("chain_type", "unknown")
        chain = " → ".join(kd.get("chain") or [])

        if all(k in kd for k in ("delta_mean", "delta_ci_low", "delta_ci_high")):
            lines.append(
                f"- **{x}** (ATE {ate:+.3f}) | What-if: Δ ≈ {float(kd['delta_mean']):+.3f} "
                f"(CI {float(kd['delta_ci_low']):+.3f}..{float(kd['delta_ci_high']):+.3f})"
            )
        else:
            lines.append(f"- **{x}** (ATE {ate:+.3f})")

        lines.append(f"  - Chain ({chain_type}): {chain}")

    lines.append("")
    lines.append("### What not to optimize (and why)")
    for item in (narr.get("do_not_optimize") or []):
        lines.append(f"- **{item['variable']}** — {item['category']}. {item['reason']}")

    lines.append("")
    lines.append("### Next steps")
    for s in (narr.get("next_steps") or []):
        lines.append(f"- {s}")

    return "\n".join(lines)


# Backward-compatible entry point used by kairos_core.py
def generate_exec_narrative(
    result: Dict[str, Any],
    y: str,
    max_drivers: int = 3,
    *,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
) -> str:
    narrator = GeminiNarrator(model=model, temperature=temperature, max_output_tokens=2500)
    narr_json = llm_exec_narrative(result=result, y=y, narrator=narrator, max_drivers=max_drivers)
    return narrative_json_to_markdown(narr_json)
