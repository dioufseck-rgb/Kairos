# =========================
# Kairos Core (PC+FCI + DoWhy + EconML + Driver-Specific SCM + YAML runner)
# =========================

from __future__ import annotations

import os
import json
import time
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

import yaml

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci

from dowhy import CausalModel
from dowhy.gcm import StructuralCausalModel, auto, fit, interventional_samples

from econml.dml import LinearDML

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def kprint(msg: str) -> None:
    print(f"[Kairos] {msg}", flush=True)


class stage:
    def __init__(self, name: str):
        self.name = name
        self.t0: Optional[float] = None

    def __enter__(self):
        self.t0 = time.time()
        kprint(f"▶ {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = (time.time() - self.t0) if self.t0 else 0.0
        if exc is None:
            kprint(f"✔ {self.name} (done in {dt:.1f}s)")
            return False
        kprint(f"✖ {self.name} (failed in {dt:.1f}s): {exc}")
        return False


@dataclass
class KairosConfig:
    # discovery
    pc_alpha: float = 0.05
    fci_alpha: float = 0.05
    min_rows_discovery: int = 200
    disc_min_unique: int = 3
    disc_min_std: float = 1e-8
    disc_max_corr: float = 0.9995

    # estimation
    econml_cv: int = 5
    min_rows_estimation: int = 50

    # difference-making
    max_drivers: int = 5
    min_abs_effect: float = 1e-9

    # SCM
    scm_samples: int = 1000
    min_rows_scm: int = 80

    # misc
    verbose: bool = True


# -------------------------
# Preprocessing
# -------------------------

def preprocess_for_estimation(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for c in work.columns:
        if work[c].dtype == "object":
            coerced = pd.to_numeric(work[c], errors="coerce")
            if coerced.notna().mean() > 0.90:
                work[c] = coerced
    return work.dropna(axis=0, how="any")


def encode_covariates(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    cols = [c for c in cols if c in df.columns]
    if len(cols) == 0:
        return np.zeros((len(df), 0), dtype=float), []

    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]

    pipe = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline(
                steps=[
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
                ]
            ), cat),
        ],
        remainder="drop",
    )
    X = pipe.fit_transform(df[cols])

    feature_names: List[str] = []
    feature_names.extend(num)
    for c in cat:
        feature_names.append(f"{c}[*]")
    return np.asarray(X, dtype=float), feature_names


# -------------------------
# Discovery-safe numeric frame (Fisher-Z needs non-singular corr)
# -------------------------

def discovery_ready_frame(
    df: pd.DataFrame,
    cols: List[str],
    *,
    min_unique: int,
    min_std: float,
    max_corr: float,
    verbose: bool,
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    work = df[cols].copy()
    dropped: Dict[str, List[str]] = {"non_numeric": [], "low_unique": [], "low_std": [], "high_corr": []}

    keep: List[str] = []
    for c in cols:
        if c not in work.columns:
            continue
        if pd.api.types.is_numeric_dtype(work[c]):
            keep.append(c)
        else:
            coerced = pd.to_numeric(work[c], errors="coerce")
            if coerced.notna().mean() > 0.95:
                work[c] = coerced
                keep.append(c)
            else:
                dropped["non_numeric"].append(c)

    work = work[keep].dropna(axis=0, how="any")

    nunique = work.nunique(dropna=True)
    low_unique_cols = nunique[nunique < min_unique].index.tolist()
    if low_unique_cols:
        dropped["low_unique"].extend(low_unique_cols)
        work = work.drop(columns=low_unique_cols)

    stds = work.std(numeric_only=True)
    low_std_cols = stds[stds < min_std].index.tolist()
    if low_std_cols:
        dropped["low_std"].extend(low_std_cols)
        work = work.drop(columns=low_std_cols)

    if work.shape[1] >= 2:
        corr = work.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            if (upper[col] > max_corr).any():
                to_drop.add(col)
        if to_drop:
            dropped["high_corr"].extend(sorted(to_drop))
            work = work.drop(columns=list(to_drop))

    kept_cols = work.columns.tolist()

    if verbose:
        kprint("[Discovery] discovery_ready_frame summary")
        kprint(f"  requested cols : {len(cols)}")
        kprint(f"  kept cols      : {len(kept_cols)}")
        kprint(f"  rows used      : {len(work)}")
        for k, v in dropped.items():
            if v:
                kprint(f"  dropped {k:10s}: {len(v)} -> {v[:10]}{'...' if len(v) > 10 else ''}")

    return work, kept_cols, dropped


def _call_fci_compat(X: np.ndarray, *, alpha: float, indep_test: str, node_names: List[str]):
    sig = inspect.signature(fci)
    if "dataset" in sig.parameters:
        return fci(dataset=X, alpha=alpha, indep_test=indep_test, node_names=node_names)
    if "data" in sig.parameters:
        return fci(data=X, alpha=alpha, indep_test=indep_test, node_names=node_names)
    return fci(X, alpha, indep_test, node_names=node_names)


def _collect_edges(res: Any, feature_names: List[str], src: str) -> List[Tuple[str, str, str]]:
    mat = None
    if hasattr(res, "G") and hasattr(res.G, "graph"):
        mat = res.G.graph
    elif isinstance(res, tuple) and len(res) > 0:
        cg = res[0]
        if hasattr(cg, "G") and hasattr(cg.G, "graph"):
            mat = cg.G.graph
        elif hasattr(cg, "graph"):
            mat = cg.graph
    if mat is None:
        raise TypeError(f"Could not extract adjacency matrix from {src} result type={type(res)}")

    edges: List[Tuple[str, str, str]] = []
    for i, a in enumerate(feature_names):
        for j, b in enumerate(feature_names):
            if i == j:
                continue
            if mat[i, j] != 0:
                edges.append((a, b, src))
    return edges


def discover_pc_fci(df: pd.DataFrame, cols: List[str], cfg: KairosConfig) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    work, kept_cols, dropped = discovery_ready_frame(
        df, cols,
        min_unique=int(cfg.disc_min_unique),
        min_std=float(cfg.disc_min_std),
        max_corr=float(cfg.disc_max_corr),
        verbose=bool(cfg.verbose),
    )

    if len(work) < int(cfg.min_rows_discovery):
        raise ValueError(f"Too few rows for discovery: {len(work)} (min={cfg.min_rows_discovery})")
    if len(kept_cols) < 3:
        raise ValueError("Too few usable columns for discovery after cleaning (need >=3).")

    X = work[kept_cols].to_numpy(dtype=float)
    feature_names = kept_cols

    kprint(f"[Discovery] Running PC on {X.shape[0]}x{X.shape[1]} (alpha={cfg.pc_alpha})")
    pc_res = pc(data=X, alpha=float(cfg.pc_alpha), indep_test="fisherz", node_names=feature_names)

    kprint(f"[Discovery] Running FCI on {X.shape[0]}x{X.shape[1]} (alpha={cfg.fci_alpha})")
    fci_res = _call_fci_compat(X, alpha=float(cfg.fci_alpha), indep_test="fisherz", node_names=feature_names)

    edges = _collect_edges(pc_res, feature_names, "pc") + _collect_edges(fci_res, feature_names, "fci")
    disc_info = {"kept_cols": kept_cols, "dropped": dropped, "rows_used": int(len(work))}
    return edges, disc_info


def build_dag(edges: List[Tuple[str, str, str]], cols: List[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(cols)
    for a, b, src in edges:
        if a not in G or b not in G:
            continue
        if not G.has_edge(a, b):
            G.add_edge(a, b, source=src)
        else:
            cur = str(G[a][b].get("source", ""))
            if src not in cur.split(","):
                G[a][b]["source"] = (cur + "," + src).strip(",")

    while True:
        try:
            cyc = nx.find_cycle(G)
            G.remove_edge(*cyc[0])
        except Exception:
            break
    return G


# -------------------------
# DoWhy + EconML effect estimation
# -------------------------

def estimate_effect(df: pd.DataFrame, dag: nx.DiGraph, x: str, y: str, confounders: List[str], cfg: KairosConfig) -> Dict[str, Any]:
    if len(df) < int(cfg.min_rows_estimation):
        raise ValueError(f"Too few rows for estimation: {len(df)} (min={cfg.min_rows_estimation})")

    graph_str = None
    try:
        graph_str = nx.nx_pydot.to_pydot(dag).to_string()
    except Exception:
        pass

    model = CausalModel(data=df, treatment=x, outcome=y, graph=graph_str)
    estimand = model.identify_effect()

    Y = pd.to_numeric(df[y], errors="coerce").to_numpy()
    T = pd.to_numeric(df[x], errors="coerce").to_numpy()
    X, _ = encode_covariates(df, confounders)

    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=200, random_state=0),
        model_t=RandomForestRegressor(n_estimators=200, random_state=0),
        cv=int(cfg.econml_cv),
        random_state=0,
    )
    est.fit(Y, T, X=X if X.shape[1] > 0 else None)

    ite = est.effect(X if X.shape[1] > 0 else None)
    ate_per_unit = float(np.mean(ite))

    t_series = pd.to_numeric(df[x], errors="coerce").dropna()
    t_std = float(t_series.std()) if len(t_series) else 0.0
    ate_per_1sd = float(ate_per_unit * t_std) if t_std > 0 else None

    return {
        "x": x,
        "effect": ate_per_unit,        # per +1 unit (debug)
        "effect_per_1sd": ate_per_1sd, # per +1σ (exec-facing)
        "treatment_std": t_std,
        "estimand": str(estimand),
        "method": "DoWhy identify + EconML LinearDML",
        "rows": int(len(df)),
        "confounders": list(confounders),
    }


# -------------------------
# Difference-making filter
# -------------------------

def difference_making(effects: List[Dict[str, Any]], controllable: Set[str], cfg: KairosConfig) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    supported: List[Dict[str, Any]] = []
    eliminated: Dict[str, str] = {}

    for e in effects:
        x = e.get("x")
        if not x:
            continue
        if "error" in e:
            eliminated[x] = str(e["error"])
            continue
        if x not in controllable:
            eliminated[x] = "Uncontrollable (context only)"
            continue

        eff = e.get("effect_per_1sd", None)
        if eff is None:
            eff = e.get("effect", 0.0)

        if eff is None or abs(float(eff)) < float(cfg.min_abs_effect):
            eliminated[x] = "No material causal effect"
            continue

        supported.append(e)

    supported.sort(
        key=lambda d: abs(float(d.get("effect_per_1sd") if d.get("effect_per_1sd") is not None else d.get("effect", 0.0))),
        reverse=True
    )
    return supported[: int(cfg.max_drivers)], eliminated


def extract_causal_chains(dag: nx.DiGraph, drivers: List[Dict[str, Any]], y: str, max_len: int = 6) -> Dict[str, List[Dict[str, Any]]]:
    chains: Dict[str, List[Dict[str, Any]]] = {}
    for d in drivers:
        x = d.get("x")
        if not x or x not in dag or y not in dag:
            continue
        try:
            path = nx.shortest_path(dag, source=x, target=y)
            if len(path) > max_len:
                path = path[:max_len-1] + [y]
            chains[x] = [{"chain": path, "type": "directed_shortest"}]
        except Exception:
            chains[x] = [{"chain": [x, y], "type": "fallback"}]
    return chains


# -------------------------
# SCM counterfactuals (driver-specific!)
# -------------------------

def propose_intervention_value(df: pd.DataFrame, x: str) -> Optional[float]:
    s = pd.to_numeric(df[x], errors="coerce").dropna()
    if len(s) == 0 or s.nunique() < 5:
        return None
    q50 = float(s.quantile(0.50))
    q75 = float(s.quantile(0.75))
    if abs(q75 - q50) < 1e-9:
        return None
    return q75


def scm_subgraph_for_driver(dag: nx.DiGraph, x: str, y: str) -> Optional[nx.DiGraph]:
    if x not in dag or y not in dag:
        return None
    nodes = set(nx.descendants(dag, x)) | {x, y}
    nodes |= set(dag.predecessors(y))  # stabilize Y mechanism
    return dag.subgraph(nodes).copy()


def _const_intervention(v: float):
    return lambda _: float(v)


def gcm_interventional_samples_compat(scm: StructuralCausalModel, interventions: Dict[str, Any], n_samples: int):
    sig = inspect.signature(interventional_samples)
    if "num_samples_to_draw" in sig.parameters:
        return interventional_samples(scm, interventions=interventions, num_samples_to_draw=int(n_samples))
    if "num_samples" in sig.parameters:
        return interventional_samples(scm, interventions=interventions, num_samples=int(n_samples))
    return interventional_samples(scm, interventions=interventions)


def build_and_sample_scm_driver_specific(
    df: pd.DataFrame,
    dag: nx.DiGraph,
    x: str,
    y: str,
    intervention_value: float,
    cfg: KairosConfig
) -> Dict[str, Any]:
    if not nx.has_path(dag, x, y):
        return {"error": "No directed causal path to outcome in discovered DAG; SCM what-if skipped."}

    subdag = scm_subgraph_for_driver(dag, x, y)
    if subdag is None:
        return {"error": "Failed to construct SCM subgraph."}

    nodes = list(subdag.nodes)
    work = preprocess_for_estimation(df[nodes])

    if len(work) < int(cfg.min_rows_scm):
        return {"error": f"Too few rows for SCM: {len(work)} (min={cfg.min_rows_scm})"}

    scm = StructuralCausalModel(subdag)
    auto.assign_causal_mechanisms(scm, work)
    fit(scm, work)

    interventions = {x: _const_intervention(float(intervention_value))}
    cf = gcm_interventional_samples_compat(scm, interventions=interventions, n_samples=int(cfg.scm_samples))

    y_cf = pd.to_numeric(cf[y], errors="coerce").dropna()
    y_base = pd.to_numeric(work[y], errors="coerce").dropna()
    if len(y_cf) == 0 or len(y_base) == 0:
        return {"error": "SCM produced non-numeric/empty outcome samples."}

    cf_mean = float(y_cf.mean())
    ci_low, ci_high = np.percentile(y_cf.values, [5, 95]).tolist()
    base_mean = float(y_base.mean())

    return {
        "counterfactual_mean": cf_mean,
        "counterfactual_ci": [float(ci_low), float(ci_high)],
        "interventions": {x: float(intervention_value)},
        "rows_used_for_scm": int(len(work)),
        "scm_nodes": nodes,
        "baseline_y_mean": base_mean,
        "delta_mean": float(cf_mean - base_mean),
        "delta_ci": [float(ci_low - base_mean), float(ci_high - base_mean)],
    }


# -------------------------
# Narrative (minimal)
# -------------------------

def narrative_text(y: str, selected: List[Dict[str, Any]], eliminated: Dict[str, str], counterfactuals: Dict[str, Any], causal_chains: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = [f"Kairos causal explanation for {y}.", "", "Key causal drivers (controllable only):"]
    if not selected:
        lines.append("- None identified.")
    for d in selected:
        x = d["x"]
        ate1 = d.get("effect_per_1sd", None)
        ate_str = "NA" if ate1 is None else f"{float(ate1):+.3f} (per +1σ)"
        lines.append(f"- {x}: ATE={ate_str} (rows={d.get('rows','NA')})")

    lines.append("")
    lines.append("SCM counterfactual scenarios:")
    if not counterfactuals:
        lines.append("- None.")
    for x, cf in counterfactuals.items():
        if isinstance(cf, dict) and "error" in cf:
            lines.append(f"- do({x}) => ERROR: {cf['error']}")
        elif isinstance(cf, dict) and "delta_mean" in cf:
            dm = float(cf["delta_mean"])
            lo, hi = cf.get("delta_ci", [None, None])
            lines.append(f"- do({x}) => Δ{y} ≈ {dm:+.3f} (CI {lo:+.3f}..{hi:+.3f})")
        else:
            lines.append(f"- do({x}) => {cf}")

    lines.append("")
    lines.append("Causal chains:")
    for d in selected:
        x = d["x"]
        chain = causal_chains.get(x, [])
        if chain:
            lines.append(f"- {x}: {' → '.join(chain[0]['chain'])} ({chain[0].get('type','')})")
        else:
            lines.append(f"- {x}: {x} → {y}")

    lines.append("")
    lines.append("Not causally relevant / not estimable:")
    for k, v in eliminated.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


# -------------------------
# Orchestrator
# -------------------------

def run_kairos(df: pd.DataFrame, y: str, explanans: List[str], controllable: List[str], uncontrollable: List[str], cfg: KairosConfig) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    controllable_set = set(controllable)

    with stage("Initialize"):
        kprint(f"Outcome (explanandum): {y}")
        kprint(f"Explanans candidates provided: {len(explanans)}")
        df = preprocess_for_estimation(df)
        kprint(f"Input data: rows={len(df)}, cols={df.shape[1]}")

    cols = [c for c in ([y] + list(explanans)) if c in df.columns]

    with stage("Run causal discovery (PC + FCI)"):
        edges, disc_info = discover_pc_fci(df, cols, cfg)
        result["discovery_info"] = disc_info

    with stage("Build DAG from discovered edges"):
        dag = build_dag(edges, cols)
        result["dag"] = dag

    effects: List[Dict[str, Any]] = []
    with stage("Estimate causal effects (DoWhy + EconML)"):
        for x in explanans:
            if x not in df.columns or x == y or x not in controllable_set:
                continue
            confounders = list(set(uncontrollable) & set(dag.predecessors(x)))
            try:
                eff = estimate_effect(df, dag, x, y, confounders, cfg)
                effects.append(eff)
                if cfg.verbose:
                    kprint(f"  • {x}: ate_per_1σ={eff.get('effect_per_1sd', None)}")
            except Exception as e:
                effects.append({"x": x, "error": str(e)})
                if cfg.verbose:
                    kprint(f"  • {x}: FAILED ({e})")

    result["effects"] = effects

    with stage("Difference-making filter"):
        selected, eliminated = difference_making(effects, set(controllable), cfg)
        result["selected_drivers"] = selected
        result["eliminated"] = eliminated
        kprint(f"Selected drivers: {len(selected)} / {len(effects)}")

    with stage("Extract causal chains"):
        causal_chains = extract_causal_chains(dag, selected, y=y, max_len=6)
        result["causal_chains"] = causal_chains

    counterfactuals: Dict[str, Any] = {}
    with stage(f"SCM counterfactuals for {len(selected)} selected drivers"):
        for i, d in enumerate(selected, start=1):
            x = d["x"]
            kprint(f"  • SCM {i}/{len(selected)}: do({x})")

            intervention_value = propose_intervention_value(df, x)
            if intervention_value is None:
                counterfactuals[x] = {"error": "SCM skipped: non-numeric / low-variance / discrete intervention variable."}
                kprint("    → SKIPPED: no valid intervention value")
                continue

            s = pd.to_numeric(df[x], errors="coerce").dropna()
            if len(s):
                kprint(f"    {x} stats: mean={s.mean():.4f} std={s.std():.4f} p50={s.quantile(0.5):.4f} p75={s.quantile(0.75):.4f}")
            kprint(f"    do({x}) set-to: {intervention_value}")

            t0 = time.time()
            cf = build_and_sample_scm_driver_specific(df, dag, x, y, float(intervention_value), cfg)
            counterfactuals[x] = cf
            if "error" in cf:
                kprint(f"    → FAILED: {cf['error']} (in {time.time()-t0:.1f}s)")
            else:
                kprint(f"    → Δ{y}={cf['delta_mean']:+.3f} CI={cf['delta_ci']} (in {time.time()-t0:.1f}s)")

    result["counterfactuals"] = counterfactuals

    with stage("Write minimal narrative"):
        result["narrative"] = narrative_text(y, selected, eliminated, counterfactuals, causal_chains)

    return result


# -------------------------
# YAML runner
# -------------------------

def build_kairos_config_from_yaml(cfgd: Dict[str, Any]) -> KairosConfig:
    cfgd = cfgd or {}
    allowed = set(inspect.signature(KairosConfig).parameters.keys())
    filtered = {k: cfgd[k] for k in cfgd.keys() if k in allowed}
    unknown = sorted([k for k in cfgd.keys() if k not in allowed])
    if unknown:
        kprint(f"YAML config keys ignored (not in KairosConfig): {unknown}")

    def _as_int(v): return int(v) if v is not None else v
    def _as_float(v): return float(v) if v is not None else v
    def _as_bool(v):
        if isinstance(v, bool): return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true","yes","y","1"): return True
            if s in ("false","no","n","0"): return False
        return bool(v)

    for k in ("pc_alpha","fci_alpha","disc_min_std","disc_max_corr","min_abs_effect"):
        if k in filtered: filtered[k] = _as_float(filtered[k])

    for k in ("min_rows_discovery","disc_min_unique","econml_cv","min_rows_estimation","max_drivers","scm_samples","min_rows_scm"):
        if k in filtered: filtered[k] = _as_int(filtered[k])

    if "verbose" in filtered:
        filtered["verbose"] = _as_bool(filtered["verbose"])

    return KairosConfig(**filtered)


def run_from_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    dataset = spec.get("dataset") or {}
    path = dataset.get("path")
    if not path:
        raise ValueError("YAML dataset.path is required")

    with stage("Load dataset"):
        df = pd.read_csv(path)
        kprint(f"Loaded {path}: rows={len(df)}, cols={df.shape[1]}")

    types = spec.get("types") or {}
    numeric_cols = set(types.get("numeric") or [])
    if numeric_cols:
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    target = spec.get("target") or {}
    y = target.get("outcome")
    if not y:
        raise ValueError("YAML target.outcome is required")

    roles = spec.get("roles") or {}
    explanans = roles.get("explanans") or []
    controllable = roles.get("controllable") or []
    uncontrollable = roles.get("uncontrollable") or []

    if not explanans:
        explanans = [c for c in df.columns if c != y and c not in set(uncontrollable)]

    def _missing(cols): return [c for c in cols if c not in df.columns]
    for name, cols in (("roles.explanans", explanans), ("roles.controllable", controllable), ("roles.uncontrollable", uncontrollable)):
        missing = _missing(cols)
        if missing:
            raise ValueError(f"YAML {name} contains missing columns: {missing}")

    cfg = build_kairos_config_from_yaml(spec.get("config") or {})

    result = run_kairos(df=df, y=y, explanans=explanans, controllable=controllable, uncontrollable=uncontrollable, cfg=cfg)

    llm_spec = spec.get("llm") or {}
    if bool(llm_spec.get("enabled", False)):
        with stage("LLM executive narrative"):
            try:
                from kairos_narrative import generate_exec_narrative
                exec_text = generate_exec_narrative(
                    result=result,
                    y=y,
                    max_drivers=int(getattr(cfg, "max_drivers", 5)),
                    model=llm_spec.get("model", "gemini-2.5-flash"),
                    temperature=float(llm_spec.get("temperature", 0.2)),
                )
                result["exec_narrative"] = exec_text
            except Exception as e:
                result["exec_narrative_error"] = str(e)
                kprint(f"LLM narrative failed: {e}")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        kprint("Usage: python kairos_core.py /path/to/kairos_run.yaml")
        raise SystemExit(2)
    out = run_from_yaml(sys.argv[1])
    print("\n" + "="*80 + "\n")
    print(out.get("narrative", ""))
