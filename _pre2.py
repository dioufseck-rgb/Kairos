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
import numpy as np
import pandas as pd

def discovery_ready_frame(
    df: pd.DataFrame,
    cols: list[str],
    *,
    min_unique: int = 3,
    min_std: float = 1e-8,
    max_corr: float = 0.9995,
    verbose: bool = True,
):
    """
    Prepare numeric matrix for Fisher-Z CI tests used by PC/FCI.

    Steps:
    1) Keep only numeric cols (Fisher-Z requires numeric)
    2) Drop constant / near-constant cols
    3) Drop cols with too few unique values
    4) Drop one of any pair with extremely high correlation (near-collinearity)

    Returns:
      df_clean, kept_cols, dropped_info
    """
    work = df[cols].copy()

    # Ensure numeric only (coerce if needed)
    dropped = {"non_numeric": [], "low_unique": [], "low_std": [], "high_corr": []}

    keep = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(work[c]):
            keep.append(c)
        else:
            # try coercion
            coerced = pd.to_numeric(work[c], errors="coerce")
            if coerced.notna().mean() > 0.95:
                work[c] = coerced
                keep.append(c)
            else:
                dropped["non_numeric"].append(c)

    work = work[keep].dropna(axis=0, how="any")

    # Drop too-few-unique
    nunique = work.nunique(dropna=True)
    low_unique_cols = nunique[nunique < min_unique].index.tolist()
    if low_unique_cols:
        dropped["low_unique"].extend(low_unique_cols)
        work = work.drop(columns=low_unique_cols)

    # Drop near-constant (std ~ 0)
    stds = work.std(numeric_only=True)
    low_std_cols = stds[stds < min_std].index.tolist()
    if low_std_cols:
        dropped["low_std"].extend(low_std_cols)
        work = work.drop(columns=low_std_cols)

    # Drop near-collinear columns
    # (Greedy: for any pair with |corr|>max_corr, drop the latter column.)
    if work.shape[1] >= 2:
        corr = work.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            # if any correlation above threshold, drop this col
            if (upper[col] > max_corr).any():
                to_drop.add(col)
        if to_drop:
            dropped["high_corr"].extend(sorted(to_drop))
            work = work.drop(columns=list(to_drop))

    kept_cols = work.columns.tolist()

    if verbose:
        print("[Discovery] discovery_ready_frame summary")
        print(f"  requested cols: {len(cols)}")
        print(f"  kept cols     : {len(kept_cols)}")
        for k, v in dropped.items():
            if v:
                print(f"  dropped {k}: {len(v)} -> {v[:10]}{'...' if len(v)>10 else ''}")
        print(f"  rows used      : {len(work)}")

    return work, kept_cols, dropped



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

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci

def discover_pc_fci(df: pd.DataFrame, cols: list[str], cfg):
    # Prepare a Fisher-Z safe matrix
    work, kept_cols, dropped = discovery_ready_frame(
        df, cols,
        min_unique=getattr(cfg, "disc_min_unique", 3),
        min_std=getattr(cfg, "disc_min_std", 1e-8),
        max_corr=getattr(cfg, "disc_max_corr", 0.9995),
        verbose=getattr(cfg, "verbose", True),
    )

    if len(work) < getattr(cfg, "min_rows_discovery", 200):
        raise ValueError(f"Too few rows for discovery: {len(work)} (min={getattr(cfg,'min_rows_discovery',200)})")

    if len(kept_cols) < 3:
        raise ValueError("Too few usable columns for discovery after cleaning (need >=3).")

    X = work[kept_cols].to_numpy(dtype=float)
    feature_names = kept_cols

    print(f"[Discovery] Running PC on {X.shape[0]}x{X.shape[1]} (fisherz alpha={cfg.pc_alpha})")
    pc_res = pc(data=X, alpha=cfg.pc_alpha, indep_test="fisherz", node_names=feature_names)

    print(f"[Discovery] Running FCI on {X.shape[0]}x{X.shape[1]} (fisherz alpha={cfg.fci_alpha})")
    #fci_res = fci(data=X, alpha=cfg.fci_alpha, indep_test="fisherz", node_names=feature_names)
    fci_res = fci(dataset=X, alpha=cfg.fci_alpha, indep_test="fisherz", node_names=feature_names)

    def collect_edges(res, src: str):
        # causallearn returns different structures depending on algo; pc_res has .G; fci returns (graph, sep_sets, ...)
        G = getattr(res, "G", None)
        if G is None and isinstance(res, tuple):
            # fci commonly returns (graph, edges, ...)
            for item in res:
                if hasattr(item, "graph"):
                    G = item
                    break
        if G is None:
            # Another common pattern: tuple first element is causal graph object
            if isinstance(res, tuple) and len(res) > 0 and hasattr(res[0], "G"):
                G = res[0].G

        # In causallearn PC, pc_res.G.graph is adjacency matrix
        # In fci, the first element is usually a CausalGraph with .G.graph
        if hasattr(res, "G") and hasattr(res.G, "graph"):
            mat = res.G.graph
        elif isinstance(res, tuple) and hasattr(res[0], "graph"):
            mat = res[0].graph
        elif isinstance(res, tuple) and hasattr(res[0], "G") and hasattr(res[0].G, "graph"):
            mat = res[0].G.graph
        else:
            # last resort: try res.G.graph
            mat = res.G.graph

        edges = []
        for i, a in enumerate(feature_names):
            for j, b in enumerate(feature_names):
                if i == j:
                    continue
                if mat[i, j] != 0:
                    edges.append((a, b, src))
        return edges

    pc_edges = collect_edges(pc_res, "pc")
    fci_edges = collect_edges(fci_res, "fci")

    # Merge and return
    edges = pc_edges + fci_edges

    # Also return dropped info so you can log it or include in result
    return edges, {"kept_cols": kept_cols, "dropped": dropped}



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
        y_num = pd.to_numeric(df[y], errors="coerce").dropna()
        if len(y_num) == 0:
            baseline_y_mean = None
            baseline_y_ci = None
            kprint(f"Baseline {y}: not numeric / empty after coercion (baseline deltas disabled).")
        else:
            baseline_y_mean = float(y_num.mean())
            baseline_y_ci = np.percentile(y_num.values, [5, 95]).tolist()
            kprint(
                f"Baseline {y}: mean={baseline_y_mean:.3f}, "
                f"CI5-95=[{baseline_y_ci[0]:.3f}, {baseline_y_ci[1]:.3f}]"
            )

    result: Dict[str, Any] = {}

    if y not in df.columns:
        raise ValueError(f"Outcome column not found: {y}")

    cols = [y] + [c for c in explanans if c in df.columns and c != y]

    if cfg.verbose:
        shown = cols[:min(len(cols), cfg.max_print_cols)]
        kprint(f"Variables used (y + explanans present): {len(cols)} -> {shown}{' ...' if len(cols) > cfg.max_print_cols else ''}")

    # --- Discovery
    with stage("Run causal discovery (PC + FCI)"):
        edges, disc_info = discover_pc_fci(df, cols, cfg)
        result["discovery_info"] = disc_info

    with stage("Build DAG from discovered edges"):
        dag = build_dag(edges, cols, cfg)
        result["dag"] = dag


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
                    # --- Convert absolute counterfactual into delta on outcome (if baseline available)
                    cf_mean = float(counterfactuals[x]["counterfactual_mean"])
                    ci_low, ci_high = counterfactuals[x]["counterfactual_ci"]

                    if baseline_y_mean is not None:
                        delta_mean = cf_mean - baseline_y_mean
                        delta_ci = [ci_low - baseline_y_mean, ci_high - baseline_y_mean]

                        counterfactuals[x]["delta_mean"] = delta_mean
                        counterfactuals[x]["delta_ci"] = delta_ci
                        counterfactuals[x]["baseline_y_mean"] = baseline_y_mean  # handy for cards

                        kprint(f"    → Δ{y}={delta_mean:+.4f} CI=[{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}] (in {time.time()-t0:.1f}s)")
                    else:
                        kprint(f"    → CF mean={cf_mean:.4f} CI=[{ci_low:.4f}, {ci_high:.4f}] (baseline unavailable) (in {time.time()-t0:.1f}s)")

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

import yaml

def load_run_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_types(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    t = spec.get("types") or {}
    num = set(t.get("numeric") or [])
    cat = set(t.get("categorical") or [])

    df = df.copy()
    for c in df.columns:
        if c in num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif c in cat:
            df[c] = df[c].astype("object")
    return df

import inspect

def build_kairos_config_from_yaml(cfgd: Dict[str, Any]) -> KairosConfig:
    """Build KairosConfig from YAML while ignoring unknown keys safely.

    Also coerces common numeric/bool fields to the correct type to avoid runtime
    type errors (e.g., comparing int to str).
    """
    cfgd = cfgd or {}
    allowed = set(inspect.signature(KairosConfig).parameters.keys())

    filtered = {k: cfgd[k] for k in cfgd.keys() if k in allowed}
    unknown = sorted([k for k in cfgd.keys() if k not in allowed])
    if unknown:
        print(f"[Kairos] YAML config keys ignored (not in KairosConfig): {unknown}")

    def _as_int(v, default=None):
        try:
            if v is None:
                return default
            return int(v)
        except Exception:
            return default

    def _as_float(v, default=None):
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _as_bool(v, default=None):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "yes", "y", "1", "on"):
                return True
            if s in ("false", "no", "n", "0", "off"):
                return False
        return default

    # ---- floats
    for k in ("pc_alpha", "fci_alpha", "min_abs_effect", "disc_min_std", "disc_max_corr"):
        if k in filtered:
            filtered[k] = _as_float(filtered[k], filtered[k])

    # ---- ints
    for k in (
        "econml_cv", "max_drivers", "scm_samples",
        "min_rows_discovery", "min_rows_estimation", "min_rows_scm",
        "disc_min_unique",
        "max_print_cols",
    ):
        if k in filtered:
            filtered[k] = _as_int(filtered[k], filtered[k])

    # ---- bools
    for k in ("verbose",):
        if k in filtered:
            filtered[k] = _as_bool(filtered[k], filtered[k])

    return KairosConfig(**filtered)


# =========================
# YAML Runner
# =========================

import inspect
from typing import Optional
import yaml


def load_run_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid YAML spec (expected dict) in {path}")
    return spec


def apply_types(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply explicit typing from YAML spec:
      types:
        numeric: [...]
        categorical: [...]
    Any column not listed is left as-is (later steps may infer/encode).
    """
    t = spec.get("types") or {}
    numeric = set(t.get("numeric") or [])
    categorical = set(t.get("categorical") or [])

    work = df.copy()
    for c in work.columns:
        if c in numeric:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        elif c in categorical:
            work[c] = work[c].astype("object")
    return work


def preprocess_from_yaml(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Uses your existing preprocess_for_estimation(df) if present; otherwise
    does a minimal preprocessing guided by YAML.
    """
    pp = spec.get("preprocess", {}) or {}
    work = df.copy()

    # Optional trim
    if pp.get("trim_whitespace_in_strings", True):
        for c in work.columns:
            if work[c].dtype == "object":
                work[c] = work[c].astype(str).str.strip()

    # Optional numeric coercion on object columns
    thr = float(pp.get("coerce_numeric_threshold", 0.90))
    for c in work.columns:
        if work[c].dtype == "object":
            coerced = pd.to_numeric(work[c], errors="coerce")
            if coerced.notna().mean() >= thr:
                work[c] = coerced

    # Drop rows with NA (if requested)
    if pp.get("drop_rows_with_any_na", False):
        work = work.dropna(axis=0, how="any")

    # If your codebase already has preprocess_for_estimation, prefer it.
    if "preprocess_for_estimation" in globals() and callable(globals()["preprocess_for_estimation"]):
        work = globals()["preprocess_for_estimation"](work)

    return work


def build_kairos_config_from_yaml(cfgd: Dict[str, Any]) -> KairosConfig:
    """
    Build KairosConfig from YAML while ignoring unknown keys safely.
    Prevents TypeError when YAML contains keys not present in KairosConfig.
    """
    allowed = set(inspect.signature(KairosConfig).parameters.keys())
    cfgd = cfgd or {}

    filtered = {k: cfgd[k] for k in cfgd.keys() if k in allowed}
    unknown = sorted([k for k in cfgd.keys() if k not in allowed])
    if unknown:
        print(f"[Kairos] YAML config keys ignored (not in KairosConfig): {unknown}")

    # Convert basic types where needed (defensive)
    # Only coerce keys that commonly appear.
    if "pc_alpha" in filtered:
        filtered["pc_alpha"] = float(filtered["pc_alpha"])
    if "fci_alpha" in filtered:
        filtered["fci_alpha"] = float(filtered["fci_alpha"])
    if "econml_cv" in filtered:
        filtered["econml_cv"] = int(filtered["econml_cv"])
    if "max_drivers" in filtered:
        filtered["max_drivers"] = int(filtered["max_drivers"])
    if "scm_samples" in filtered:
        filtered["scm_samples"] = int(filtered["scm_samples"])
    if "verbose" in filtered:
        filtered["verbose"] = bool(filtered["verbose"])

    return KairosConfig(**filtered)


def run_from_yaml(spec_path: str) -> Dict[str, Any]:
    """
    End-to-end runner:
      - Load YAML
      - Load CSV
      - Apply include/exclude
      - Apply types + preprocessing
      - Build KairosConfig from YAML (filtered)
      - Run causal core (run_kairos)
      - Optionally generate LLM exec narrative (from YAML llm section)

    Expected YAML keys (minimal):
      dataset.path
      target.outcome
      roles.controllable / roles.uncontrollable
      config (optional)
      llm (optional)
    """
    spec = load_run_spec(spec_path)

    # ---------------------------
    # 1) Load dataset
    # ---------------------------
    ds = spec.get("dataset") or {}
    csv_path = ds.get("path")
    if not csv_path:
        raise ValueError("YAML missing dataset.path (CSV file path).")

    encoding = ds.get("encoding", "utf-8")
    df = pd.read_csv(csv_path, encoding=encoding)

    # ---------------------------
    # 2) Outcome (explanandum)
    # ---------------------------
    tgt = spec.get("target") or {}
    y = tgt.get("outcome")
    if not y:
        raise ValueError("YAML missing target.outcome (explanandum).")
    if y not in df.columns:
        raise ValueError(f"Outcome '{y}' not found in CSV columns.")

    print(f"[Kairos] Loaded dataset: {csv_path}  rows={len(df)} cols={df.shape[1]}")
    print(f"[Kairos] Outcome: {y}")

    # ---------------------------
    # 3) include/exclude columns
    # ---------------------------
    cols_spec = spec.get("columns") or {}
    include = cols_spec.get("include") or []
    exclude = set(cols_spec.get("exclude") or [])

    if include:
        missing = [c for c in include if c not in df.columns]
        if missing:
            raise ValueError(f"YAML columns.include contains missing columns: {missing}")
        df = df[include]

    if exclude:
        df = df[[c for c in df.columns if c not in exclude]]

    if y not in df.columns:
        raise ValueError("After include/exclude, outcome column is missing. Adjust YAML.")

    # ---------------------------
    # 4) Apply explicit types + preprocessing
    # ---------------------------
    df = apply_types(df, spec)
    df = preprocess_from_yaml(df, spec)

    print(f"[Kairos] After preprocessing: rows={len(df)} cols={df.shape[1]}")

    # ---------------------------
    # 5) Roles: explanans, controllable, uncontrollable
    # ---------------------------
    roles = spec.get("roles") or {}
    explanans = roles.get("explanans") or []
    controllable = roles.get("controllable") or []
    uncontrollable = roles.get("uncontrollable") or []

    # Infer explanans if not provided
    if not explanans:
        explanans = [c for c in df.columns if c != y]

    # Basic validation
    for name, lst in [("explanans", explanans), ("controllable", controllable), ("uncontrollable", uncontrollable)]:
        missing = [c for c in lst if c not in df.columns]
        if missing:
            raise ValueError(f"YAML roles.{name} contains missing columns: {missing}")

    overlap = set(controllable) & set(uncontrollable)
    if overlap:
        raise ValueError(f"Columns cannot be both controllable and uncontrollable: {sorted(list(overlap))}")

    if y in explanans:
        explanans = [c for c in explanans if c != y]

    # ---------------------------
    # 6) Build core config from YAML and run
    # ---------------------------
    cfgd = spec.get("config") or {}
    cfg = build_kairos_config_from_yaml(cfgd)

    print(f"[Kairos] Running core with: pc_alpha={getattr(cfg,'pc_alpha',None)} "
          f"fci_alpha={getattr(cfg,'fci_alpha',None)} max_drivers={getattr(cfg,'max_drivers',None)}")

    result = run_kairos(
        df=df,
        y=y,
        explanans=explanans,
        controllable=controllable,
        uncontrollable=uncontrollable,
        cfg=cfg,
    )
    import os, json, time

    def _ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

    def _write_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _write_json(path: str, obj: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _safe_make_visuals(result: dict, y: str, out_dir: str) -> dict:
        from kairos_viz import (
            make_visual_bundle,
            render_chain_cards_png,
            render_dot_to_file,           # may fail if dot missing
            render_nx_fallback_png,        # we will add this in kairos_viz.py below
        )

        _ensure_dir(out_dir)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(out_dir, f"kairos_{stamp}")

        print(f"[Viz] Output dir: {out_dir}")
        print("[Viz] Building visual bundle...")
        viz = make_visual_bundle(result=result, y=y)

        # Always write DOT files (even if PNG rendering fails)
        if viz.get("full_dot"):
            _write_text(prefix + "_full.dot", viz["full_dot"])
            print("[Viz] Wrote:", prefix + "_full.dot")
        if viz.get("drivers_dot"):
            _write_text(prefix + "_drivers.dot", viz["drivers_dot"])
            print("[Viz] Wrote:", prefix + "_drivers.dot")

        # Always write summary JSON
        _write_json(prefix + "_bundle.json", {
            "y": y,
            "selected_drivers": result.get("selected_drivers", []),
            "counterfactuals": result.get("counterfactuals", {}),
            "causal_chains": result.get("causal_chains", {}),
        })
        print("[Viz] Wrote:", prefix + "_bundle.json")

        # Try Graphviz PNGs
        try:
            if viz.get("full_dot"):
                render_dot_to_file(viz["full_dot"], prefix + "_full", fmt="png")
                print("[Viz] Wrote:", prefix + "_full.png")
            if viz.get("drivers_dot"):
                render_dot_to_file(viz["drivers_dot"], prefix + "_drivers", fmt="png")
                print("[Viz] Wrote:", prefix + "_drivers.png")
        except Exception as e:
            print(f"[Viz] Graphviz render failed ({type(e).__name__}): {e}")
            print("[Viz] Falling back to NetworkX PNG rendering...")
            # Fallback PNGs that do not require system 'dot'
            if result.get("dag") is not None:
                render_nx_fallback_png(result["dag"], prefix + "_full_fallback.png")
                print("[Viz] Wrote:", prefix + "_full_fallback.png")

        # Chain cards PNG (pure PIL; should always work)
        try:
            cards_path = render_chain_cards_png(
                y=y,
                selected_drivers=result.get("selected_drivers", []),
                causal_chains=result.get("causal_chains", {}),
                counterfactuals=result.get("counterfactuals", {}),
                baseline=result.get("baseline", {}),
                out_path=prefix + "_cards.png",
            )
            print("[Viz] Wrote:", cards_path)
        except Exception as e:
            print(f"[Viz] Chain cards failed ({type(e).__name__}): {e}")

        return {"out_prefix": prefix, "viz": viz}
    
    out_dir = (spec.get("outputs") or {}).get("dir", "/workspaces/Kairos/outputs")
    viz_info = _safe_make_visuals(result=result, y=y, out_dir=out_dir)
    result["outputs"] = viz_info
    
    # ---------------------------
    # 7) Optional LLM narrative (from YAML llm section)
    # ---------------------------
    llm_spec = spec.get("llm") or {}
    if llm_spec.get("enabled", False):
        from kairos_narrative import generate_exec_narrative

        model = llm_spec.get("model", "gemini-2.5-flash")
        temperature = float(llm_spec.get("temperature", 0.2))
        max_dr = int(llm_spec.get("max_drivers", getattr(cfg, "max_drivers", 5)))

        print(f"[Kairos] Generating exec narrative via LLM: model={model} temp={temperature} max_drivers={max_dr}")
        exec_text = generate_exec_narrative(
            result=result,
            y=y,
            max_drivers=max_dr,
            model=model,
            temperature=temperature,
        )
        result["exec_narrative"] = exec_text

    return result


# =========================
# Example usage (your main)
# =========================

if __name__ == "__main__":
    result = run_from_yaml("/workspaces/Kairos/kairos_run_manufacturing.yaml")
    #result = run_from_yaml("/workspaces/Kairos/kairos_run_cmapss_fd001.yaml")
    print("\n" + "=" * 80)
    print(result.get("narrative", ""))
    if "exec_narrative" in result:
        print("\n" + "=" * 80)
        print(result["exec_narrative"])
