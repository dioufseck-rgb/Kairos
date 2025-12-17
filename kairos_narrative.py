# kairos_narrative.py
# =========================
# Kairos Executive Narrative (Gemini-backed, hallucination-proof fields)
# - Chains: LLM must copy top_chain_text verbatim; we inject canonical chain list/text from evidence
# - Deltas: LLM outputs structured delta_mean/delta_ci; we inject canonical values from evidence
#           and we render "Δ" ourselves so validation doesn't depend on LLM formatting
# =========================

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Gemini SDK compatibility: supports either "google-genai" (new) or "google-generativeai" (old)
GENAI_BACKEND = None
try:
    from google import genai as genai_new  # pip install google-genai
    GENAI_BACKEND = "google-genai"
except Exception:
    genai_new = None

try:
    import google.generativeai as genai_old  # pip install google-generativeai
    if GENAI_BACKEND is None:
        GENAI_BACKEND = "google-generativeai"
except Exception:
    genai_old = None


# -------------------------
# JSON utilities (robust)
# -------------------------

def _extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level {...} JSON object from a response.
    Handles common wrappers like ```json ... ```
    """
    if not text:
        return ""

    # Strip code fences if present
    text2 = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    start = text2.find("{")
    if start == -1:
        return ""

    depth = 0
    for i in range(start, len(text2)):
        ch = text2[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text2[start : i + 1]

    return text2[start:]


def _json_loads_strict(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty JSON string")
    return json.loads(raw)


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _format_ci(ci: Any, ndigits: int = 3) -> Optional[Tuple[float, float]]:
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return None
    a = _as_float(ci[0])
    b = _as_float(ci[1])
    if a is None or b is None:
        return None
    return (a, b)


# -------------------------
# Gemini wrapper
# -------------------------

@dataclass
class GeminiNarrator:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    api_key_env: str = "GEMINI_KEY"

    def __post_init__(self):
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"Gemini API key env var '{self.api_key_env}' is not set.")

        if GENAI_BACKEND == "google-genai":
            self.client = genai_new.Client(api_key=key)
        elif GENAI_BACKEND == "google-generativeai":
            genai_old.configure(api_key=key)
            self.client = genai_old.GenerativeModel(self.model)
        else:
            raise RuntimeError(
                "No Gemini SDK found. Install one of:\n"
                "  pip install google-genai\n"
                "or\n"
                "  pip install google-generativeai"
            )

    def generate_text(self, prompt: str) -> str:
        if GENAI_BACKEND == "google-genai":
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": self.temperature},
            )
            txt = getattr(resp, "text", None)
            return txt or str(resp)

        # google-generativeai
        resp = self.client.generate_content(
            prompt,
            generation_config={"temperature": self.temperature},
        )
        txt = getattr(resp, "text", None)
        if txt:
            return txt
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            return str(resp)

    def generate_json(self, prompt: str, *, max_repairs: int = 2) -> Dict[str, Any]:
        raw = self.generate_text(prompt)
        blob = _extract_first_json_object(raw)

        try:
            return _json_loads_strict(blob)
        except Exception as e:
            last_raw = raw
            last_err = e

            for _ in range(max_repairs):
                repair_prompt = f"""
Return ONLY valid JSON. No markdown. No commentary.

Your previous output was not valid JSON.

PREVIOUS_OUTPUT:
{last_raw}

Return corrected JSON now:
"""
                raw2 = self.generate_text(repair_prompt)
                blob2 = _extract_first_json_object(raw2)
                try:
                    return _json_loads_strict(blob2)
                except Exception as e2:
                    last_raw = raw2
                    last_err = e2

            raise ValueError(
                "Gemini returned non-JSON after repair attempts.\n"
                f"Last parse error: {repr(last_err)}\n"
                f"RAW (truncated):\n{(last_raw or '')[:4000]}"
            )


# -------------------------
# Evidence packet
# -------------------------

def build_evidence_packet(result: Dict[str, Any], y: str, max_drivers: int = 5) -> Dict[str, Any]:
    """
    Build evidence packet for the LLM.
    Includes canonical chain + counterfactual deltas for each driver (if available).
    """
    selected = result.get("selected_drivers") or []
    eliminated = result.get("eliminated") or {}
    counterfactuals = result.get("counterfactuals") or {}
    causal_chains = result.get("causal_chains") or {}

    allowed: set[str] = {str(y)}

    for d in selected:
        allowed.add(str(d.get("x", "")))
    for k in eliminated.keys():
        allowed.add(str(k))

    # include chain nodes already discovered
    for x, chains in causal_chains.items():
        allowed.add(str(x))
        if isinstance(chains, list):
            for ch in chains:
                path = ch.get("chain")
                if isinstance(path, list):
                    for node in path:
                        allowed.add(str(node))

    allowed_variables = sorted(v for v in allowed if v and str(v).strip())

    drivers_out: List[Dict[str, Any]] = []
    for d in (selected[:max_drivers] if isinstance(selected, list) else []):
        x = str(d.get("x", ""))
        ate = d.get("effect", None)
        ate_units = d.get("ate_units", "per +1σ")
        rows = d.get("rows_used", d.get("rows", None))

        # best available chain for x
        chain_info = None
        chains = causal_chains.get(x) or []
        if isinstance(chains, list) and len(chains) > 0:
            chain_info = chains[0]

        if chain_info and isinstance(chain_info.get("chain"), list) and len(chain_info["chain"]) > 0:
            top_chain = [str(n) for n in chain_info["chain"]]
            chain_type = str(chain_info.get("type", "unknown"))
        else:
            top_chain = [x, str(y)]
            chain_type = "fallback"

        top_chain_text = " → ".join(top_chain)

        cf = counterfactuals.get(x) or {}
        delta_mean = cf.get("delta_mean", None) if isinstance(cf, dict) else None
        delta_ci = cf.get("delta_ci", None) if isinstance(cf, dict) else None
        cf_err = cf.get("error", None) if isinstance(cf, dict) else None
        scm_mode = cf.get("mode", cf.get("path_mode", cf.get("scm_mode", None))) if isinstance(cf, dict) else None

        drivers_out.append({
            "x": x,
            "ate": ate,
            "ate_units": ate_units,
            "rows_used": rows,
            "top_chain": top_chain,
            "top_chain_text": top_chain_text,
            "chain_type": chain_type,
            "delta_mean": delta_mean,
            "delta_ci": delta_ci,
            "counterfactual_error": cf_err,
            "scm_mode": scm_mode,
        })

    packet = {
        "y": str(y),
        "allowed_variables": allowed_variables,
        "drivers": drivers_out,
        "eliminated": {str(k): str(v) for k, v in eliminated.items()},
        "notes": {
            "ate_semantics": "ATE is reported in units stated by ate_units; often standardized (per +1σ).",
            "counterfactual_semantics": "delta_mean/delta_ci represent change in outcome under a realistic intervention, relative to baseline.",
            "chain_semantics": "top_chain_text is the discovered/derived path to the outcome; copy verbatim in the LLM output.",
        },
    }
    return packet


def classify_non_action_reason(reason: str) -> str:
    r = (reason or "").lower()
    if "uncontroll" in r:
        return "uncontrollable_context"
    if "singular" in r or "matrix" in r:
        return "data_issue"
    if "not estimable" in r or "identif" in r:
        return "weak_identification"
    if "no material" in r or "no effect" in r:
        return "weak_effect"
    return "other"


# -------------------------
# Validation + canonical injection
# -------------------------

def validate_narrative_json(narr: Dict[str, Any], evidence: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    allowed = set(evidence.get("allowed_variables") or [])
    eliminated = set((evidence.get("eliminated") or {}).keys())

    if not isinstance(narr, dict):
        return False, ["Narrative JSON is not an object"]

    for k in ["title", "executive_summary", "key_drivers", "do_not_optimize", "next_steps"]:
        if k not in narr:
            issues.append(f"Missing required field '{k}'")

    kds = narr.get("key_drivers")
    if not isinstance(kds, list):
        issues.append("key_drivers must be a list")
    else:
        for i, kd in enumerate(kds):
            if not isinstance(kd, dict):
                issues.append(f"key_drivers[{i}] must be an object")
                continue

            var = kd.get("variable")
            if var not in allowed:
                issues.append(f"key_drivers[{i}].variable='{var}' not in allowed_variables.")

            ct = kd.get("chain_text")
            if not isinstance(ct, str) or not ct.strip():
                issues.append(f"key_drivers[{i}].chain_text missing/empty")
            else:
                ev = None
                for d in evidence.get("drivers") or []:
                    if d.get("x") == var:
                        ev = d
                        break
                if ev is not None:
                    expected = ev.get("top_chain_text")
                    if expected and ct.strip() != str(expected).strip():
                        issues.append(
                            f"key_drivers[{i}].chain_text must exactly match evidence top_chain_text for '{var}'."
                        )

            wi = kd.get("what_if")
            required_phrase = "If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected"
            if not isinstance(wi, str) or required_phrase not in wi:
                issues.append(f"key_drivers[{i}].what_if missing required framing language")

            # Structured delta fields
            if "counterfactual_available" not in kd:
                issues.append(f"key_drivers[{i}].counterfactual_available missing")
            else:
                avail = kd.get("counterfactual_available")
                if not isinstance(avail, bool):
                    issues.append(f"key_drivers[{i}].counterfactual_available must be boolean")
                elif avail is True:
                    dm = kd.get("delta_mean", None)
                    dci = kd.get("delta_ci", None)
                    if _as_float(dm) is None:
                        issues.append(f"key_drivers[{i}].delta_mean must be a number when counterfactual_available=true")
                    if not isinstance(dci, list) or len(dci) != 2 or _as_float(dci[0]) is None or _as_float(dci[1]) is None:
                        issues.append(f"key_drivers[{i}].delta_ci must be [low, high] numbers when counterfactual_available=true")

    dno = narr.get("do_not_optimize")
    if not isinstance(dno, list):
        issues.append("do_not_optimize must be a list")
    else:
        for i, item in enumerate(dno):
            if not isinstance(item, dict):
                issues.append(f"do_not_optimize[{i}] must be an object")
                continue
            var = item.get("variable")
            if var not in allowed:
                issues.append(f"do_not_optimize[{i}].variable='{var}' not in allowed_variables.")
            if eliminated and var not in eliminated:
                issues.append(f"do_not_optimize[{i}].variable='{var}' must come from eliminated variables only.")
            cat = item.get("category")
            if cat not in {"uncontrollable_context", "data_issue", "weak_identification", "weak_effect", "other"}:
                issues.append(f"do_not_optimize[{i}].category invalid")

    ns = narr.get("next_steps")
    if not isinstance(ns, list) or not all(isinstance(x, str) for x in ns):
        issues.append("next_steps must be a list of strings")

    return (len(issues) == 0), issues


def inject_canonical_fields(narr: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overwrite canonical chain + delta fields from evidence (hallucination-proof).
    """
    ev_by_x = {d["x"]: d for d in (evidence.get("drivers") or []) if isinstance(d, dict) and d.get("x")}

    kds = narr.get("key_drivers") or []
    if isinstance(kds, list):
        for kd in kds:
            if not isinstance(kd, dict):
                continue
            x = kd.get("variable")
            ev = ev_by_x.get(x)
            if not ev:
                continue

            # canonical chain
            kd["chain"] = ev.get("top_chain")
            kd["chain_text"] = ev.get("top_chain_text")
            kd["chain_type"] = ev.get("chain_type")
            kd["scm_mode"] = ev.get("scm_mode")

            # canonical delta fields
            dm = ev.get("delta_mean", None)
            dci = ev.get("delta_ci", None)
            ok_ci = _format_ci(dci) is not None
            ok_dm = _as_float(dm) is not None
            kd["delta_mean"] = dm if ok_dm else None
            kd["delta_ci"] = dci if ok_ci else [None, None]
            kd["counterfactual_available"] = bool(ok_dm and ok_ci and not ev.get("counterfactual_error"))

            # carry error if any
            if ev.get("counterfactual_error"):
                kd["counterfactual_error"] = ev.get("counterfactual_error")

    return narr


# -------------------------
# LLM Prompt + Orchestration
# -------------------------

def llm_exec_narrative(
    result: Dict[str, Any],
    y: str,
    narrator: GeminiNarrator,
    max_drivers: int = 5,
) -> Dict[str, Any]:
    evidence = build_evidence_packet(result, y=y, max_drivers=max_drivers)

    eliminated = evidence.get("eliminated") or {}
    do_not_seed = [{
        "variable": str(var),
        "category": classify_non_action_reason(str(reason)),
        "reason": str(reason),
    } for var, reason in eliminated.items()]

    prompt = f"""
You are Kairos, an executive explanation writer for causal findings.

HARD RULES (NON-NEGOTIABLE)
1) Use ONLY variables in allowed_variables.
2) Use ONLY numbers that appear in the EVIDENCE JSON (ATEs, delta_mean, delta_ci, rows).
3) Do NOT invent causal mechanisms. ONLY describe pathways using the provided top_chain_text.
4) For EACH driver, you MUST copy the driver’s "top_chain_text" EXACTLY into "chain_text". No edits. No extra nodes.
5) Counterfactual framing MUST use this exact language (verbatim):
   "If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected {y} change, given the discovered causal structure?"
6) Counterfactual output MUST be structured:
   - delta_mean: number or null (copy evidence delta_mean)
   - delta_ci: [low, high] numbers or [null, null] (copy evidence delta_ci)
   - counterfactual_available: boolean (true only if evidence has valid delta_mean and delta_ci and no counterfactual_error)
   Do NOT output a text delta summary.
7) Include "What not to optimize (and why)" grounded ONLY in eliminated variables using DO_NOT_OPTIMIZE_SEED.
   Do NOT introduce new eliminated variables.
8) Round any numbers in the text summary to 3 decimal places.

OUTPUT FORMAT
Return VALID JSON ONLY (no markdown, no commentary) with this structure:

{{
  "title": "string",
  "executive_summary": "string",
  "key_drivers": [
    {{
      "variable": "one of allowed_variables",
      "ate_summary": "string (include ATE from evidence; do not invent numbers)",
      "what_if": "string (must include the required counterfactual framing sentence verbatim)",
      "delta_mean": "number or null",
      "delta_ci": ["number or null", "number or null"],
      "counterfactual_available": "boolean",
      "chain_text": "string (MUST match evidence top_chain_text exactly)",
      "uncertainty_notes": "string (mention uncertainty; if scm_mode indicates salvaged/local/undirected, say direction not uniquely identified)"
    }}
  ],
  "do_not_optimize": [
    {{
      "variable": "must come from eliminated variables only",
      "category": "one of: uncontrollable_context | data_issue | weak_identification | weak_effect | other",
      "reason": "string (must match eliminated reason; no invented reasons)"
    }}
  ],
  "next_steps": ["string", "string", "..."]
}}

EVIDENCE (JSON):
{json.dumps(evidence, ensure_ascii=False)}

DO_NOT_OPTIMIZE_SEED (JSON):
{json.dumps(do_not_seed, ensure_ascii=False)}
"""

    narr = narrator.generate_json(prompt)

    ok, issues = validate_narrative_json(narr, evidence)
    if not ok:
        repair_prompt = f"""
Fix ONLY these violations. Do not add any new variables. Return VALID JSON ONLY.

VIOLATIONS:
{json.dumps(issues, ensure_ascii=False, indent=2)}

EVIDENCE (JSON):
{json.dumps(evidence, ensure_ascii=False)}

PREVIOUS_JSON:
{json.dumps(narr, ensure_ascii=False)}
"""
        narr2 = narrator.generate_json(repair_prompt)
        ok2, issues2 = validate_narrative_json(narr2, evidence)
        if not ok2:
            raise ValueError("Narrative validation failed after repair:\n" + "\n".join(issues2))
        narr = narr2

    # Overwrite canonical chain + deltas from evidence (hallucination-proof)
    narr = inject_canonical_fields(narr, evidence)
    return narr


# -------------------------
# Public API: render executive narrative text
# -------------------------

def render_exec_narrative_text(narr_json: Dict[str, Any], y: str) -> str:
    """
    Turn the narrative JSON into the readable exec report text.
    This is where we render "Δ" consistently (not dependent on LLM formatting).
    """
    title = narr_json.get("title", f"Causal Analysis Report: {y}")
    summary = (narr_json.get("executive_summary") or "").strip()
    key_drivers = narr_json.get("key_drivers") or []
    dno = narr_json.get("do_not_optimize") or []
    steps = narr_json.get("next_steps") or []

    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    if summary:
        lines.append(summary)
        lines.append("")

    lines.append("### Key causal drivers")
    if not key_drivers:
        lines.append("- None identified.")
    else:
        for kd in key_drivers:
            var = kd.get("variable", "")
            ate = (kd.get("ate_summary") or "").strip()
            wi = (kd.get("what_if") or "").strip()
            ct = (kd.get("chain_text") or "").strip()
            un = (kd.get("uncertainty_notes") or "").strip()

            # structured deltas
            avail = bool(kd.get("counterfactual_available", False))
            dm = kd.get("delta_mean", None)
            dci = kd.get("delta_ci", None)

            lines.append(f"- **{var}** — {ate}".strip())

            if wi:
                lines.append(f"  - {wi}")

            if avail and _as_float(dm) is not None and _format_ci(dci) is not None:
                dm_f = float(dm)
                ci_low, ci_high = _format_ci(dci)  # type: ignore[misc]
                lines.append(f"  - What-if: Δ ≈ {dm_f:+.3f} (CI {ci_low:+.3f}..{ci_high:+.3f})")
            else:
                err = kd.get("counterfactual_error")
                if err:
                    lines.append(f"  - What-if: counterfactual not available ({err})")
                else:
                    lines.append("  - What-if: counterfactual not available")

            if ct:
                lines.append(f"  - Chain: {ct}")

            if un:
                lines.append(f"  - Notes: {un}")

    lines.append("")
    lines.append("### What not to optimize (and why)")
    if not dno:
        lines.append("- None.")
    else:
        for item in dno:
            v = item.get("variable", "")
            cat = item.get("category", "")
            reason = item.get("reason", "")
            lines.append(f"- **{v}** — {cat}. {reason}".strip())

    lines.append("")
    lines.append("### Next steps")
    if not steps:
        lines.append("- Define validation experiments and monitoring for selected drivers.")
    else:
        for s in steps:
            s2 = str(s).strip()
            if s2:
                lines.append(f"- {s2}")

    return "\n".join(lines).strip()


def generate_exec_narrative(
    result: Dict[str, Any],
    y: str,
    max_drivers: int = 5,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    api_key_env: str = "GEMINI_KEY",
) -> str:
    """
    Main function called from kairos_core.py.
    Returns a formatted executive narrative (text).
    """
    narrator = GeminiNarrator(model=model, temperature=temperature, api_key_env=api_key_env)
    narr_json = llm_exec_narrative(result=result, y=y, narrator=narrator, max_drivers=max_drivers)
    return render_exec_narrative_text(narr_json=narr_json, y=y)
