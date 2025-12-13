# =========================
# Kairos Core (PC+FCI + DoWhy + EconML + SCM) — with Progress Indicators
# =========================
# Key robustness + UX upgrades:
# - NaN-safe discovery matrix (impute + OHE)
# - causal-learn PC/FCI API + return-type variance (tuple-safe)
# - prevents one-hot sibling edges from leaking into column-level graph
# - suppresses noisy causal-learn stdout (edges printed during discovery)
# - DoWhy graph passed as nx.DiGraph (no DOT parsing issues)
# - EconML uses W for confounders, X=None (avoids empty-feature crashes)
# - SCM uses numeric-only nodes (avoids DoWhy GCM classification/logloss pitfalls)
# - interventional_samples compatibility across DoWhy versions:
#   * interventions as callables (float -> lambda)
#   * avoids "observed_samples AND num_samples_to_draw" conflict by subsampling observed_data
# - Progress prints + timings (flush=True) so execution feels responsive
# =========================

import sys
import io
import time
import inspect
import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Any, Set, Tuple, Optional
from contextlib import contextmanager

# --- Discovery (causal-learn)
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci

# --- Causal inference
from dowhy import CausalModel
from dowhy.gcm import StructuralCausalModel, auto, fit, interventional_samples

# --- EconML
from econml.dml import LinearDML

# --- ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# =========================
# Progress / logging helpers
# =========================

def kprint(msg: str):
    print(f"[Kairos] {msg}", flush=True)


@contextmanager
def stage(name: str):
    kprint(f"▶ {name} ...")
    t0 = time.time()
    try:
        yield
        kprint(f"✔ {name} (done in {time.time() - t0:.1f}s)")
    except Exception as e:
        kprint(f"✖ {name} (failed in {time.time() - t0:.1f}s): {e}")
        raise


@contextmanager
def suppress_stdout_stderr():
    """Silence libs that print to stdout/stderr (causal-learn can be noisy)."""
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# =========================
# Config
# =========================

@dataclass
class KairosConfig:
    # Discovery
    pc_alpha: float = 0.05
    fci_alpha: float = 0.05
    discovery_drop_first_ohe: bool = True
    min_rows_discovery: int = 200

    # Effect estimation
    econml_cv: int = 5
    min_rows_estimation: int = 200

    # Difference-making
    max_drivers: int = 5
    min_abs_effect: float = 1e-6

    # SCM
    scm_samples: int = 1000
    min_rows_scm: int = 200
    intervention_numeric_delta: float = 1.0
    clamp_likert_min: float = 1.0
    clamp_likert_max: float = 5.0
    likert_detection_threshold: float = 0.95  # >=95% values within [1,5] => clamp

    # UX
    verbose: bool = True
    print_dag_summary: bool = True
    max_print_cols: int = 25  # avoid huge spam


# =========================
# Utilities
# =========================

def preprocess_for_estimation(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning; keep NaNs and handle per-step. Coerce mostly-numeric object columns."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.9:
                df[c] = coerced
    return df


def _to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")


def _binary_encode_treatment(work: pd.DataFrame, x: str) -> Tuple[pd.Series, Optional[Dict[Any, int]]]:
    """
    For EconML: if x is categorical/object, map to binary using top-2 categories.
    This allows LinearDML discrete_treatment=True in those cases.
    """
    s = work[x]
    if pd.api.types.is_numeric_dtype(s):
        return _to_numeric_series(s), None

    vc = s.value_counts(dropna=True)
    if len(vc) < 2:
        return pd.Series([np.nan] * len(work), index=work.index), None

    top2 = vc.index[:2].tolist()
    mapping = {top2[0]: 0, top2[1]: 1}
    return s.map(mapping), mapping


def encode_controls_W(work: pd.DataFrame, cols: List[str]) -> Tuple[Optional[np.ndarray], Optional[ColumnTransformer]]:
    """Encode confounders as W (controls) for EconML. Return (None,None) if empty."""
    if not cols:
        return None, None

    num = [c for c in cols if pd.api.types.is_numeric_dtype(work[c])]
    cat = [c for c in cols if c not in num]

    transformers = []
    if num:
        transformers.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
            ]),
            cat
        ))

    pipe = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    W = pipe.fit_transform(work[cols])
    return np.asarray(W, dtype=float), pipe


def dag_to_dot(dag: nx.DiGraph, highlight_y: Optional[str] = None) -> str:
    def esc(s: str) -> str:
        return s.replace('"', '\\"')

    lines = ["digraph Kairos {", '  rankdir="LR";']
    for n in dag.nodes():
        if highlight_y and n == highlight_y:
            lines.append(f'  "{esc(n)}" [shape=doubleoctagon, style=filled, fillcolor="#fef08a"];')
        else:
            lines.append(f'  "{esc(n)}" [shape=box, style=rounded];')
    for u, v, d in dag.edges(data=True):
        src = d.get("source", "")
        lines.append(f'  "{esc(u)}" -> "{esc(v)}" [label="{esc(src)}"];')
    lines.append("}")
    return "\n".join(lines)


# =========================
# Discovery preprocessing (numeric, no NaNs)
# =========================

def _build_discovery_matrix(
    df: pd.DataFrame,
    cols: List[str],
    cfg: KairosConfig
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    sub = df[cols].copy()

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(sub[c])]
    cat_cols = [c for c in cols if c not in num_cols]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(
            handle_unknown="ignore",
            drop=("first" if cfg.discovery_drop_first_ohe else None),
            sparse_output=False
        )),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    with stage("Discovery preprocessing (impute + encode)"):
        X = np.asarray(pre.fit_transform(sub), dtype=float)

    if np.isnan(X).any():
        raise ValueError("Discovery matrix contains NaNs after imputation/encoding.")

    try:
        feature_names = pre.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"v{i}" for i in range(X.shape[1])]

    # Robust mapping: match exact or "col_" prefix (not partial)
    feat_to_col: Dict[str, str] = {}
    for fn in feature_names:
        mapped = None
        for c in cols:
            if fn == c or fn.startswith(c + "_"):
                mapped = c
                break
        feat_to_col[fn] = mapped if mapped is not None else fn

    if cfg.verbose:
        kprint(f"Discovery matrix X shape: {X.shape} (features={len(feature_names)})")
        # Show a few feature names only
        preview = feature_names[:min(len(feature_names), 15)]
        kprint(f"Encoded feature preview: {preview}{' ...' if len(feature_names) > 15 else ''}")

    return X, feature_names, feat_to_col


def _extract_graph_matrix(res: Any) -> np.ndarray:
    """Handle causal-learn return-type variance (sometimes tuple)."""
    if hasattr(res, "G") and hasattr(res.G, "graph"):
        return res.G.graph
    if hasattr(res, "graph"):
        return res.graph
    if isinstance(res, tuple):
        for item in res:
            if hasattr(item, "G") and hasattr(item.G, "graph"):
                return item.G.graph
            if hasattr(item, "graph"):
                return item.graph
    raise TypeError(f"Cannot extract graph matrix from result type: {type(res)}")


# =========================
# Discovery: PC + FCI
# =========================

def discover_pc_fci(df: pd.DataFrame, cols: List[str], cfg: KairosConfig) -> List[Tuple[str, str, str]]:
    with stage("Causal discovery setup"):
        X, feature_names, feat_to_col = _build_discovery_matrix(df, cols, cfg)

    if X.shape[0] < cfg.min_rows_discovery:
        raise ValueError(f"Too few rows for discovery: {X.shape[0]} (min={cfg.min_rows_discovery})")

    with stage("PC algorithm"):
        with suppress_stdout_stderr():
            pc_res = pc(data=X, alpha=cfg.pc_alpha, indep_test="fisherz", node_names=feature_names)
        pc_mat = _extract_graph_matrix(pc_res)

    with stage("FCI algorithm (can be slow)"):
        with suppress_stdout_stderr():
            fci_res = fci(dataset=X, independence_test_method="fisherz", alpha=cfg.fci_alpha, node_names=feature_names)
        fci_mat = _extract_graph_matrix(fci_res)

    def collect_edges(mat: np.ndarray) -> List[Tuple[str, str]]:
        """
        Collect feature-level edges, map to raw-column edges, and
        forbid edges between one-hot siblings from same raw column.
        """
        out: List[Tuple[str, str]] = []
        n = len(feature_names)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if mat[i, j] == 0:
                    continue
                a_feat, b_feat = feature_names[i], feature_names[j]
                a_col, b_col = feat_to_col[a_feat], feat_to_col[b_feat]

                # Critical: eliminate within-column one-hot edges
                if a_col == b_col:
                    continue

                out.append((a_col, b_col))
        return out

    with stage("Edge aggregation (feature -> raw columns)"):
        pc_edges = collect_edges(pc_mat)
        fci_edges = collect_edges(fci_mat)

        merged: Dict[Tuple[str, str], Set[str]] = {}
        for (a, b) in pc_edges:
            merged.setdefault((a, b), set()).add("pc")
        for (a, b) in fci_edges:
            merged.setdefault((a, b), set()).add("fci")

        edges = [(a, b, ",".join(sorted(srcs))) for (a, b), srcs in merged.items()]

    if cfg.verbose:
        kprint(f"Discovered raw edges: {len(edges)}")
        if len(edges) > 0:
            preview = edges[:min(len(edges), 12)]
            kprint(f"Edge preview: {preview}{' ...' if len(edges) > 12 else ''}")

    return edges


def build_dag(edges: List[Tuple[str, str, str]], cols: List[str], cfg: KairosConfig) -> nx.DiGraph:
    with stage("Build DAG (and break cycles)"):
        G = nx.DiGraph()
        G.add_nodes_from(cols)

        for a, b, src in edges:
            if a == b:
                continue
            if not G.has_edge(a, b):
                G.add_edge(a, b, source=src)

        # Break cycles conservatively
        removed = 0
        while True:
            try:
                cycle = nx.find_cycle(G, orientation="original")
                u, v, _ = cycle[0]
                G.remove_edge(u, v)
                removed += 1
            except Exception:
                break

    if cfg.verbose and cfg.print_dag_summary:
        kprint(f"DAG summary: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, cycle_edges_removed={removed}")
    return G


# =========================
# DoWhy + EconML effect estimation
# =========================

def estimate_effect(
    df_all_nodes: pd.DataFrame,
    dag: nx.DiGraph,
    x: str,
    y: str,
    confounders: List[str],
    cfg: KairosConfig
) -> Dict[str, Any]:

    needed = [y, x] + confounders
    work = df_all_nodes[needed].copy()

    work[y] = _to_numeric_series(work[y])
    t_series, mapping = _binary_encode_treatment(work, x)
    work["_T_"] = t_series

    work = work.dropna(subset=[y, "_T_"])
    if len(work) < cfg.min_rows_estimation:
        return {"x": x, "effect": 0.0, "error": f"Too few rows after cleaning: {len(work)}"}

    # DoWhy identification data: include all dag nodes (best effort) to avoid warnings
    model_data = df_all_nodes[[n for n in dag.nodes() if n in df_all_nodes.columns]].copy()
    model_data[y] = _to_numeric_series(model_data[y])
    if mapping is not None:
        model_data[x] = model_data[x].map(mapping)
    model_data = model_data.dropna(subset=[y, x])

    # Identify effect with DoWhy
    model = CausalModel(data=model_data, treatment=x, outcome=y, graph=dag)
    estimand = model.identify_effect()

    # EconML estimation with W controls, X=None
    W, _ = encode_controls_W(work, confounders)
    Y = work[y].values.astype(float)
    T = work["_T_"].values.astype(float)

    uniq = np.unique(T)
    is_binary = len(uniq) <= 2 and set(uniq) <= {0.0, 1.0}

    model_y = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=10)
    model_t = LogisticRegression(max_iter=2000) if is_binary else RandomForestRegressor(
        n_estimators=250, random_state=42, min_samples_leaf=10
    )

    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=is_binary,
        cv=cfg.econml_cv,
        random_state=42
    )

    est.fit(Y, T, X=None, W=W)
    ate = float(est.ate(X=None))

    return {
        "x": x,
        "effect": ate,
        "estimand": str(estimand),
        "method": "DoWhy identify + EconML LinearDML (W=confounders, X=None)",
        "confounders_used": confounders,
        "treatment_mapping": mapping,
        "rows_used": len(work),
    }


# =========================
# SCM Counterfactuals (DoWhy GCM)
# =========================

def _detect_likert(series: pd.Series, cfg: KairosConfig) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False
    within = s.between(cfg.clamp_likert_min, cfg.clamp_likert_max).mean()
    return within >= cfg.likert_detection_threshold and s.nunique() <= 10


def propose_intervention_value(df: pd.DataFrame, x: str, cfg: KairosConfig) -> Optional[float]:
    if x not in df.columns:
        return None
    s = df[x]
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return None
    m = float(np.nanmean(s))
    val = m + cfg.intervention_numeric_delta
    if _detect_likert(s, cfg):
        val = float(np.clip(val, cfg.clamp_likert_min, cfg.clamp_likert_max))
    return float(val)


def _as_callable_interventions(interventions: Dict[str, float]) -> Dict[str, Any]:
    """
    Some DoWhy versions require interventions[node] to be callable:
      interventions[node](pre_value) -> post_value
    Convert numeric constants to constant functions.
    """
    out: Dict[str, Any] = {}
    for k, v in interventions.items():
        if callable(v):
            out[k] = v
        else:
            out[k] = (lambda _pre, vv=float(v): vv)
    return out


def gcm_interventional_samples_compat(
    scm: StructuralCausalModel,
    observed_data: pd.DataFrame,
    interventions: Dict[str, Any],
    n_samples: int,
    cfg: KairosConfig
) -> pd.DataFrame:
    """
    Compatibility layer for dowhy.gcm.whatif.interventional_samples.

    Your DoWhy version raises:
      "Either observed_samples or num_samples_to_draw need to be set, not both!"

    Therefore:
      - If observed_data is required/used: SUBSAMPLE observed_data to n_samples and call WITHOUT num_samples_to_draw.
      - Otherwise: call with num_samples_to_draw (or num_samples) only.
    """
    def _sample_obs(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) >= n:
            return df.sample(n=n, replace=False, random_state=42)
        return df.sample(n=n, replace=True, random_state=42)

    sig = None
    try:
        sig = inspect.signature(interventional_samples)
        params = sig.parameters
        names = list(params.keys())
    except Exception:
        params = {}
        names = []

    if cfg.verbose:
        kprint(f"SCM: interventional_samples signature params={names[:8]}{' ...' if len(names) > 8 else ''}")

    # Case A: observed_samples/observed_data is positional second argument
    if sig is not None and len(names) >= 3 and names[1] in ("observed_samples", "observed_data", "data"):
        obs = _sample_obs(observed_data, n_samples)
        return interventional_samples(scm, obs, interventions)

    # Case B: keyword observed_samples / observed_data exists
    if "observed_samples" in params:
        obs = _sample_obs(observed_data, n_samples)
        return interventional_samples(scm, observed_samples=obs, interventions=interventions)

    if "observed_data" in params:
        obs = _sample_obs(observed_data, n_samples)
        return interventional_samples(scm, observed_data=obs, interventions=interventions)

    # Case C: no observed_samples path — use explicit n if supported
    if "num_samples_to_draw" in params:
        return interventional_samples(scm, interventions=interventions, num_samples_to_draw=n_samples)

    if "num_samples" in params:
        return interventional_samples(scm, interventions=interventions, num_samples=n_samples)

    # Fallbacks
    try:
        obs = _sample_obs(observed_data, n_samples)
        return interventional_samples(scm, obs, interventions)
    except TypeError:
        return interventional_samples(scm, interventions=interventions)


def build_and_sample_scm(
    df: pd.DataFrame,
    dag: nx.DiGraph,
    y: str,
    interventions: Dict[str, float],
    cfg: KairosConfig
) -> Dict[str, Any]:

    with stage("SCM: prepare numeric-only dataset"):
        nodes = [n for n in dag.nodes() if n in df.columns]
        work = df[nodes].copy()

        # force numeric (categoricals become NaN; SCM only runs on numeric nodes)
        for c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

        numeric_cols = []
        for c in work.columns:
            s = work[c]
            if s.notna().sum() >= cfg.min_rows_scm and s.nunique(dropna=True) >= 2:
                numeric_cols.append(c)

        if cfg.verbose:
            kprint(f"SCM: candidate nodes={len(nodes)}, numeric_usable_nodes={len(numeric_cols)}")

        if y not in numeric_cols:
            return {"error": f"SCM skipped: outcome '{y}' not usable numeric (or too sparse)."}
        for k in interventions.keys():
            if k not in numeric_cols:
                return {"error": f"SCM skipped: intervention var '{k}' not usable numeric (or too sparse)."}

        dag_num = dag.subgraph(numeric_cols).copy()
        if len(dag_num.nodes()) < 2:
            return {"error": "SCM skipped: too few numeric nodes after filtering."}

        work_num = work[numeric_cols].dropna(axis=0, how="any")
        if len(work_num) < cfg.min_rows_scm:
            return {"error": f"Too few complete rows for SCM fit (numeric-only): {len(work_num)}"}

    if cfg.verbose:
        kprint(f"SCM: training rows={len(work_num)}, nodes={len(dag_num.nodes())}, edges={dag_num.number_of_edges()}")
        kprint(f"SCM: interventions={interventions}")

    scm = StructuralCausalModel(dag_num)

    with stage("SCM: assign causal mechanisms"):
        auto.assign_causal_mechanisms(scm, work_num)

    with stage("SCM: fit structural equations"):
        fit(scm, work_num)

    interventions_callable = _as_callable_interventions(interventions)

    with stage(f"SCM: draw interventional samples (n={cfg.scm_samples})"):
        cf = gcm_interventional_samples_compat(
            scm=scm,
            observed_data=work_num,
            interventions=interventions_callable,
            n_samples=cfg.scm_samples,
            cfg=cfg
        )

    if y not in cf.columns:
        return {"error": f"SCM samples missing outcome '{y}'."}

    return {
        "counterfactual_mean": float(cf[y].mean()),
        "counterfactual_ci": [float(np.percentile(cf[y], 5)), float(np.percentile(cf[y], 95))],
        "interventions": interventions,
        "rows_used_for_scm": len(work_num),
        "scm_nodes": list(dag_num.nodes()),
    }


# =========================
# Difference-making filter (placeholder for Strevens operationalization)
# =========================

def difference_making(effects: List[Dict[str, Any]], controllable: Set[str], cfg: KairosConfig):
    supported = []
    eliminated: Dict[str, str] = {}

    for e in effects:
        x = e.get("x")
        if not x:
            continue
        if "error" in e:
            eliminated[x] = e["error"]
            continue
        if abs(e.get("effect", 0.0)) < cfg.min_abs_effect:
            eliminated[x] = "No material causal effect"
            continue
        if x not in controllable:
            eliminated[x] = "Uncontrollable (context only)"
            continue
        supported.append(e)

    supported.sort(key=lambda r: abs(r.get("effect", 0.0)), reverse=True)
    return supported[:cfg.max_drivers], eliminated


# =========================
# Narrative (simple; you’ll later swap for Gemini narrative)
# =========================

def narrative(y: str, drivers: List[Dict[str, Any]], eliminated: Dict[str, str], counterfactuals: Dict[str, Any]) -> str:
    lines = [f"Kairos causal explanation for {y}.", "", "Key causal drivers (controllable only):"]
    if not drivers:
        lines.append("- None identified.")
    for d in drivers:
        lines.append(f"- {d['x']}: ATE={d['effect']:.3f} (rows={d.get('rows_used','?')})")

    lines.append("")
    lines.append("SCM counterfactual scenarios:")
    if not counterfactuals:
        lines.append("- None.")
    for k, v in counterfactuals.items():
        lines.append(f"- do({k}) => {v}")

    lines.append("")
    lines.append("Not causally relevant / not estimable:")
    if not eliminated:
        lines.append("- None")
    else:
        for k, v in eliminated.items():
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)


# =========================
# Orchestrator
# =========================

def run_kairos(
    df: pd.DataFrame,
    y: str,
    explanans: List[str],
    controllable: List[str],
    uncontrollable: List[str],
    cfg: KairosConfig
) -> Dict[str, Any]:

    with stage("Initialize"):
        kprint(f"Outcome (explanandum): {y}")
        kprint(f"Explanans candidates provided: {len(explanans)}")
        df = preprocess_for_estimation(df)
        kprint(f"Input data: rows={len(df)}, cols={df.shape[1]}")

        # --- Baseline outcome stats (for counterfactual deltas)
        baseline_y_mean = float(df[y].mean())
        baseline_y_ci = np.percentile(df[y].values, [5, 95]).tolist()
        kprint(f"Baseline {y}: mean={baseline_y_mean:.3f}, CI5-95=[{baseline_y_ci[0]:.3f}, {baseline_y_ci[1]:.3f}]")


    if y not in df.columns:
        raise ValueError(f"Outcome column not found: {y}")

    cols = [y] + [c for c in explanans if c in df.columns and c != y]

    if cfg.verbose:
        shown = cols[:min(len(cols), cfg.max_print_cols)]
        kprint(f"Variables used (y + explanans present): {len(cols)} -> {shown}{' ...' if len(cols) > cfg.max_print_cols else ''}")

    # --- Discovery
    with stage("Run causal discovery (PC + FCI)"):
        edges = discover_pc_fci(df, cols, cfg)

    dag = build_dag(edges, cols, cfg)

    controllable_set = set([c for c in controllable if c in df.columns])
    uncontrollable_set = set([c for c in uncontrollable if c in df.columns])

    if cfg.verbose:
        kprint(f"Controllable vars present: {len(controllable_set)}")
        kprint(f"Uncontrollable vars present: {len(uncontrollable_set)}")

    # --- Effect estimation
    effects: List[Dict[str, Any]] = []
    candidates = [x for x in cols if x != y and x in controllable_set]
    with stage(f"Estimate causal effects (EconML) for {len(candidates)} controllable variables"):
        for i, x in enumerate(candidates, start=1):
            kprint(f"  • Effect {i}/{len(candidates)}: estimating '{x}' -> '{y}'")

            parents_x = set(dag.predecessors(x))
            parents_y = set(dag.predecessors(y))
            confounders = sorted(list((parents_x | parents_y) & uncontrollable_set))

            if cfg.verbose:
                kprint(f"    Confounders: {confounders if confounders else 'None'}")

            t0 = time.time()
            try:
                eff = estimate_effect(df_all_nodes=df, dag=dag, x=x, y=y, confounders=confounders, cfg=cfg)
                effects.append(eff)
                if "error" in eff:
                    kprint(f"    → FAILED: {eff['error']} (in {time.time()-t0:.1f}s)")
                else:
                    kprint(f"    → ATE={eff['effect']:.4f} (rows={eff.get('rows_used','?')}) (in {time.time()-t0:.1f}s)")
            except Exception as e:
                effects.append({"x": x, "effect": 0.0, "error": str(e)})
                kprint(f"    → EXCEPTION: {e} (in {time.time()-t0:.1f}s)")

    # --- Difference-making selection
    with stage("Apply difference-making filter (Strevens placeholder)"):
        selected, eliminated = difference_making(effects, controllable_set, cfg)
        kprint(f"Selected drivers: {len(selected)} (max={cfg.max_drivers})")
        if cfg.verbose and selected:
            kprint(f"Top drivers: {[d['x'] for d in selected]}")

    # --- SCM counterfactuals
    counterfactuals: Dict[str, Any] = {}
    with stage(f"SCM counterfactuals for {len(selected)} selected drivers"):
        for i, d in enumerate(selected, start=1):
            x = d["x"]
            kprint(f"  • SCM {i}/{len(selected)}: do({x}) counterfactual")

            intervention_value = propose_intervention_value(df, x, cfg)
            if intervention_value is None:
                counterfactuals[x] = {"error": "SCM skipped: non-numeric (or empty) intervention variable."}
                kprint(f"    → SKIPPED: non-numeric/empty intervention")
                continue

            kprint(f"    Intervention value proposed: {intervention_value}")

            t0 = time.time()
            try:
                counterfactuals[x] = build_and_sample_scm(
                df=df,
                dag=dag,
                y=y,
                interventions={x: intervention_value},
                cfg=cfg
                )

                if "error" in counterfactuals[x]:
                    kprint(f"    → FAILED: {counterfactuals[x]['error']} (in {time.time()-t0:.1f}s)")
                else:
                    # --- Convert absolute counterfactual into delta on outcome
                    cf_mean = float(counterfactuals[x]["counterfactual_mean"])
                    ci_low, ci_high = counterfactuals[x]["counterfactual_ci"]

                    delta_mean = cf_mean - baseline_y_mean
                    delta_ci = [ci_low - baseline_y_mean, ci_high - baseline_y_mean]

                    counterfactuals[x]["delta_mean"] = delta_mean
                    counterfactuals[x]["delta_ci"] = delta_ci
                    counterfactuals[x]["baseline_y_mean"] = baseline_y_mean  # optional, but handy for cards

                    kprint(
                        f"    → Δ{y}={delta_mean:+.4f} "
                        f"CI=[{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}] "
                        f"(baseline={baseline_y_mean:.4f}, in {time.time()-t0:.1f}s)"
                    )

            except Exception as e:
                counterfactuals[x] = {"error": str(e)}
                kprint(f"    → EXCEPTION: {e} (in {time.time()-t0:.1f}s)")

    # --- Narrative
    with stage("Compose narrative"):
        text = narrative(y, selected, eliminated, counterfactuals)

    return {
        "dag": dag,
        "dag_dot": dag_to_dot(dag, highlight_y=y),
        "effects": effects,
        "selected_drivers": selected,
        "eliminated": eliminated,
        "counterfactuals": counterfactuals,
        "narrative": text
    }


# =========================
# Example usage (your main)
# =========================

if __name__ == "__main__":
    df = pd.read_csv("/workspaces/Kairos/synth_kairos.csv")  # replace

    y = "Overall Satisfaction"

    uncontrollable = [
        "Region", "Customer Segment", "Age Range", "Season", "Distance (Miles)"
    ]

    controllable = [
        "Total Fare Amount",
        "Communication About Status",
        "Cleanliness",
        "Comfort",
        "Wi-Fi",
        "Food & Beverage",
        # You can choose whether to treat these as controllable:
        "Departure Delay (Minutes)",
        "Arrival Delay (Minutes)",
        "On-time Performance",
        "Staffing Level (Proxy)",
        "Track Congestion (Proxy)",
    ]

    explanans = [c for c in df.columns if c != y]


    result = run_kairos(
        df=df,
        y=y,
        explanans=explanans,
        controllable=controllable,
        uncontrollable=uncontrollable,
        cfg=KairosConfig(verbose=True)
    )

    print("\n" + "=" * 80)
    print(result["narrative"])
    # print(result["dag_dot"])

    from kairos_viz import make_visual_bundle



    viz = make_visual_bundle(result, y="Overall Trip Satisfaction")
    from kairos_viz import render_dot_to_file

    render_dot_to_file(viz["full_dot"], "/workspaces/Kairos/outputs/test_full", fmt="png")
    render_dot_to_file(viz["abstract_dot"], "/workspaces/Kairos/outputs/test_abs", fmt="png")
    

    from kairos_narrative import generate_exec_narrative

    exec_narrative = generate_exec_narrative(
    result=result,
    y="Overall Trip Satisfaction",
    max_drivers=3,
)

    print(exec_narrative)
    result["exec_narrative"] = exec_narrative

    from narrative import GeminiNarrator, llm_exec_narrative, narrative_json_to_markdown

    narrator = GeminiNarrator(
    api_key= os.environ.get('GEMINI_KEY'),
    model="gemini-2.5-flash",   # or your Gemini 3 model name when available in your account
    temperature=0.2,
)

    narr_json = llm_exec_narrative(result=result, y="Overall Trip Satisfaction", narrator=narrator)
    print(narrative_json_to_markdown(narr_json))

