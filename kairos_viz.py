# =========================
# Visualization helpers (DOT + optional renders + abstraction graph + chain cards PNG)
# =========================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import os
import textwrap

import networkx as nx
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# -------------------------
# Small utilities
# -------------------------

def _escape_dot(s: str) -> str:
    return str(s).replace('"', '\\"')


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# -------------------------
# Explanatory abstraction (Strevens-style)
# -------------------------

def build_explanatory_abstract_dag(
    dag: nx.DiGraph,
    y: str,
    selected_drivers: List[Dict[str, Any]],
) -> nx.DiGraph:
    """
    Build an explanatory (Strevens-style) abstract graph.

    Semantics:
      - every selected driver must connect to Y
      - if there is any path in the discovered graph, label as inferred causal path
      - otherwise label as difference-making (effect-based)
    """
    G = nx.DiGraph()
    G.add_node(y)

    UG = dag.to_undirected()

    for d in selected_drivers:
        x = d.get("x")
        if not x:
            continue

        G.add_node(x)

        has_path = False
        try:
            if x in UG.nodes and y in UG.nodes:
                has_path = nx.has_path(UG, x, y)
        except Exception:
            has_path = False

        label = "inferred causal path" if has_path else "difference-making (effect-based)"
        G.add_edge(x, y, source=label)

    return G


# -------------------------
# Causal chain extraction (prefer longer chains when available)
# -------------------------

def extract_causal_chains(
    dag: nx.DiGraph,
    y: str,
    selected_drivers: List[Dict[str, Any]],
    max_chain_length: int = 5,
    max_chains_per_driver: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract causal chains from selected drivers to y.

    Preference order:
      1) directed paths (dag)                -> type: discovered (directed)
      2) undirected paths (dag.to_undirected)-> type: discovered (undirected)
      3) fallback [x, y]                     -> type: abstracted (difference-making)

    Within each driver, chains are sorted by length DESC, so the first chain is the longest available.
    """
    chains: Dict[str, List[Dict[str, Any]]] = {}
    UG = dag.to_undirected()

    for d in selected_drivers:
        x = d.get("x")
        if not x:
            continue

        driver_chains: List[Dict[str, Any]] = []

        # 1) Directed paths
        try:
            if x in dag.nodes and y in dag.nodes:
                for p in nx.all_simple_paths(dag, source=x, target=y, cutoff=max_chain_length):
                    driver_chains.append(
                        {"chain": p, "type": "discovered (directed)", "length": len(p) - 1}
                    )
        except Exception:
            pass

        # 2) Undirected paths (if no directed path)
        if not driver_chains:
            try:
                if x in UG.nodes and y in UG.nodes:
                    for p in nx.all_simple_paths(UG, source=x, target=y, cutoff=max_chain_length):
                        driver_chains.append(
                            {"chain": p, "type": "discovered (undirected)", "length": len(p) - 1}
                        )
            except Exception:
                pass

        # 3) Fallback
        if not driver_chains:
            driver_chains.append(
                {"chain": [x, y], "type": "abstracted (difference-making)", "length": 1}
            )

        driver_chains.sort(key=lambda c: c["length"], reverse=True)
        chains[x] = driver_chains[:max_chains_per_driver]

    return chains


# -------------------------
# DOT rendering (Graphviz optional)
# -------------------------

def dag_to_dot_styled(
    dag: nx.DiGraph,
    y: str,
    effects: Optional[List[Dict[str, Any]]] = None,
    selected_drivers: Optional[List[Dict[str, Any]]] = None,
    title: str = "Kairos Causal Graph",
    show_edge_sources: bool = True,
) -> str:
    """
    Create a styled DOT graph. Optionally overlay:
      - effects: list of {"x":..., "effect":...}
      - selected_drivers: subset list (for highlighting)
    """
    eff_map: Dict[str, float] = {}
    if effects:
        for e in effects:
            if "x" in e and "effect" in e and "error" not in e:
                v = _safe_float(e.get("effect"), None)
                if v is not None:
                    eff_map[str(e["x"])] = v

    selected = set()
    if selected_drivers:
        selected = {d.get("x") for d in selected_drivers if d.get("x")}

    abs_vals = [abs(v) for v in eff_map.values()]
    max_abs = max(abs_vals) if abs_vals else 0.0

    def width_for(node: str) -> float:
        if node not in eff_map or max_abs <= 0:
            return 1.2
        return 1.2 + 2.8 * (abs(eff_map[node]) / max_abs)

    def node_fill(node: str) -> str:
        if node == y:
            return "#FDE68A"
        if node in selected:
            return "#DBEAFE"
        return "#F9FAFB"

    def node_border(node: str) -> float:
        if node == y:
            return 3.0
        if node in selected:
            return 2.2
        return 1.2

    lines: List[str] = []
    lines.append("digraph Kairos {")
    lines.append('  rankdir="LR";')
    lines.append('  graph [fontname="Helvetica", fontsize=18, labelloc="t"];')
    lines.append('  node  [fontname="Helvetica", fontsize=12, shape=box, style="rounded,filled"];')
    lines.append('  edge  [fontname="Helvetica", fontsize=10, arrowsize=0.8];')
    lines.append(f'  label="{_escape_dot(title)}";')

    # nodes
    for n in dag.nodes():
        n = str(n)
        fill = node_fill(n)
        penw = node_border(n)
        w = width_for(n)

        if n in eff_map:
            label = f"{n}\\nATE={eff_map[n]:.3f}"
        else:
            label = n

        lines.append(
            f'  "{_escape_dot(n)}" [label="{_escape_dot(label)}", fillcolor="{fill}", penwidth={penw:.2f}, width={w:.2f}];'
        )

    # edges
    for u, v, d in dag.edges(data=True):
        u = str(u)
        v = str(v)
        src = d.get("source", "")
        elabel = src if (show_edge_sources and src) else ""

        penw = 1.2
        color = "#6B7280"
        if v == y:
            penw = 2.2
            color = "#111827"
        if u in selected and v in selected:
            penw = max(penw, 2.0)
            color = "#1D4ED8"

        if elabel:
            lines.append(
                f'  "{_escape_dot(u)}" -> "{_escape_dot(v)}" [label="{_escape_dot(elabel)}", color="{color}", penwidth={penw:.2f}];'
            )
        else:
            lines.append(
                f'  "{_escape_dot(u)}" -> "{_escape_dot(v)}" [color="{color}", penwidth={penw:.2f}];'
            )

    lines.append("}")
    return "\n".join(lines)


def render_dot_to_svg(dot: str) -> str:
    """
    Render DOT to SVG string.
    Requires python 'graphviz' package AND system graphviz executables (dot).
    """
    try:
        import graphviz  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency: graphviz (python package). Add 'graphviz' to requirements.txt") from e

    src = graphviz.Source(dot)
    svg_bytes = src.pipe(format="svg")
    return svg_bytes.decode("utf-8")


def render_dot_to_file(dot: str, out_path: str, fmt: str = "png") -> str:
    """
    Write DOT render to a file (png/svg/pdf). Returns rendered file path.
    Requires system graphviz executables (dot). If missing, this will raise.
    """
    try:
        import graphviz  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency: graphviz (python package). Add 'graphviz' to requirements.txt") from e

    src = graphviz.Source(dot)
    rendered = src.render(filename=out_path, format=fmt, cleanup=True)
    return rendered


# -------------------------
# Result tables
# -------------------------

def effects_table(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for e in result.get("effects", []) or []:
        rows.append({
            "treatment": e.get("x"),
            "ATE": e.get("effect"),
            "rows_used": e.get("rows_used"),
            "confounders_used": ", ".join(e.get("confounders_used", []) or []),
            "method": e.get("method"),
            "error": e.get("error"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(by=["ATE"], key=lambda s: s.abs(), ascending=False, na_position="last")


def selected_drivers_table(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for d in result.get("selected_drivers", []) or []:
        rows.append({
            "driver": d.get("x"),
            "ATE": d.get("effect"),
            "rows_used": d.get("rows_used"),
            "confounders_used": ", ".join(d.get("confounders_used", []) or []),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(by=["ATE"], key=lambda s: s.abs(), ascending=False, na_position="last")


def counterfactuals_table(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    cf = result.get("counterfactuals", {}) or {}
    for k, v in cf.items():
        if isinstance(v, dict) and "error" in v:
            rows.append({"driver": k, "delta_mean": None, "ci_low": None, "ci_high": None, "error": v["error"]})
        elif isinstance(v, dict):
            # prefer deltas if available, else show absolute
            if "delta_mean" in v and "delta_ci" in v:
                ci = v.get("delta_ci", [None, None])
                rows.append({
                    "driver": k,
                    "delta_mean": v.get("delta_mean"),
                    "ci_low": ci[0] if len(ci) > 0 else None,
                    "ci_high": ci[1] if len(ci) > 1 else None,
                    "error": None
                })
            else:
                ci = v.get("counterfactual_ci", [None, None])
                rows.append({
                    "driver": k,
                    "delta_mean": None,
                    "ci_low": None,
                    "ci_high": None,
                    "error": "delta not available (only absolute counterfactual present)"
                })
        else:
            rows.append({"driver": k, "delta_mean": None, "ci_low": None, "ci_high": None, "error": "invalid counterfactual format"})
    return pd.DataFrame(rows)


def eliminated_table(result: Dict[str, Any]) -> pd.DataFrame:
    elim = result.get("eliminated", {}) or {}
    return pd.DataFrame([{"variable": k, "reason": v} for k, v in elim.items()])


# -------------------------
# Chain cards PNG (no Graphviz needed)
# -------------------------

def _default_font(size: int = 18) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _wrap_lines(text: str, width_chars: int) -> List[str]:
    lines: List[str] = []
    for para in str(text).split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width_chars))
    return lines


def render_chain_cards_png(
    y: str,
    selected_drivers: List[Dict[str, Any]],
    causal_chains: Dict[str, List[Dict[str, Any]]],
    counterfactuals: Optional[Dict[str, Any]] = None,
    baseline: Optional[Dict[str, Any]] = None,
    out_path: str = "/workspaces/Kairos/outputs/kairos_chain_cards.png",
    *,
    cards_per_row: int = 2,
    card_width: int = 720,
    card_min_height: int = 260,
    padding: int = 22,
    gap: int = 22,
    wrap_chars: int = 54,
) -> str:
    """
    Render "chain cards" as a single PNG (grid).

    Each card includes:
      - Driver name + ATE
      - Top causal chain: X -> ... -> Y (discovered or abstracted)
      - Counterfactual summary as DELTA on Y when available
    """
    if counterfactuals is None:
        counterfactuals = {}
    if baseline is None:
        baseline = {}

    baseline_y_mean = baseline.get("y_mean", None)

    title_font = _default_font(22)
    body_font = _default_font(16)
    small_font = _default_font(14)

    # Prepare card texts
    card_texts: List[str] = []
    for d in selected_drivers:
        x = d.get("x", "")
        ate = d.get("effect", None)
        ate_str = "NA" if ate is None else f"{float(ate):+.3f}"

        chains = causal_chains.get(x, [])
        if chains:
            top_chain = chains[0]  # longest available due to sorting in extract_causal_chains
            chain_path = " → ".join([str(z) for z in top_chain.get("chain", [x, y])])
            chain_type = str(top_chain.get("type", "unknown"))
        else:
            chain_path = f"{x} → {y}"
            chain_type = "abstracted (difference-making)"

        cf = counterfactuals.get(x)

        # Build What-if text (prefer delta; compute delta if possible)
        cf_text = "What-if: (not available)"
        if isinstance(cf, dict) and cf:
            if "error" in cf:
                cf_text = f"What-if: (not available) {cf['error']}"
            else:
                # Try to source a baseline either from bundle baseline or per-cf baseline
                _baseline = baseline_y_mean
                if _baseline is None:
                    _baseline = cf.get("baseline_y_mean", None)

                # Prefer delta if present
                if "delta_mean" in cf and "delta_ci" in cf:
                    dmean = float(cf["delta_mean"])
                    dci = cf.get("delta_ci", [None, None])
                    cf_text = f"What-if: Δ{y} ≈ {dmean:+.3f} (CI {dci[0]:+.3f}..{dci[1]:+.3f})"
                # Compute delta if only absolute is present and baseline is available
                elif ("counterfactual_mean" in cf and "counterfactual_ci" in cf and _baseline is not None):
                    cf_mean = float(cf["counterfactual_mean"])
                    ci_low, ci_high = cf["counterfactual_ci"]
                    dmean = cf_mean - float(_baseline)
                    dci = [ci_low - float(_baseline), ci_high - float(_baseline)]
                    cf_text = f"What-if: Δ{y} ≈ {dmean:+.3f} (CI {dci[0]:+.3f}..{dci[1]:+.3f})"
                # Fallback to absolute if that’s all we have
                elif ("counterfactual_mean" in cf and "counterfactual_ci" in cf):
                    cf_mean = float(cf["counterfactual_mean"])
                    ci = cf.get("counterfactual_ci", [None, None])
                    cf_text = f"What-if: {y} ≈ {cf_mean:.3f} (CI {ci[0]:.3f}..{ci[1]:.3f})"

        txt = (
            f"{x}\n"
            f"ATE: {ate_str}\n\n"
            f"Chain ({chain_type}):\n"
            f"{chain_path}\n\n"
            f"{cf_text}"
        )
        card_texts.append(txt)

    if not card_texts:
        img = Image.new("RGB", (900, 180), "white")
        draw = ImageDraw.Draw(img)
        draw.text((24, 24), f"No selected drivers to display for {y}.", fill="black", font=title_font)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
        return out_path

    # Measure card heights based on wrapped lines
    def measure_card_height(text: str) -> int:
        lines = _wrap_lines(text, wrap_chars)
        h_title = 28
        h_body = 22
        height = padding * 2
        for i, ln in enumerate(lines):
            height += h_title if i == 0 else (h_body if ln.strip() else 10)
        return max(card_min_height, height)

    card_heights = [measure_card_height(t) for t in card_texts]
    n_cards = len(card_texts)
    rows = (n_cards + cards_per_row - 1) // cards_per_row

    # Row heights
    row_heights: List[int] = []
    for r in range(rows):
        start = r * cards_per_row
        end = min(start + cards_per_row, n_cards)
        row_heights.append(max(card_heights[start:end]))

    img_width = cards_per_row * card_width + (cards_per_row - 1) * gap + 2 * padding
    img_height = sum(row_heights) + (rows - 1) * gap + 2 * padding + 54  # header band

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    header = f"Kairos — Causal Chain Cards (Outcome: {y})"
    draw.text((padding, padding), header, fill="black", font=title_font)
    y_cursor = padding + 44

    def draw_card(x0: int, y0: int, w: int, h: int, text: str) -> None:
        draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=18,
                               fill="#F9FAFB", outline="#111827", width=3)

        lines = _wrap_lines(text, wrap_chars)
        tx = x0 + padding
        ty = y0 + padding

        for i, ln in enumerate(lines):
            if i == 0:
                draw.text((tx, ty), ln, fill="black", font=title_font)
                ty += 30
            elif ln.startswith("ATE:"):
                draw.text((tx, ty), ln, fill="black", font=body_font)
                ty += 24
            elif ln.startswith("Chain"):
                draw.text((tx, ty), ln, fill="black", font=body_font)
                ty += 24
            elif ln.startswith("What-if"):
                draw.text((tx, ty), ln, fill="black", font=small_font)
                ty += 22
            else:
                if not ln.strip():
                    ty += 10
                else:
                    draw.text((tx, ty), ln, fill="black", font=body_font)
                    ty += 22

    idx = 0
    for r in range(rows):
        x_cursor = padding
        for _c in range(cards_per_row):
            if idx >= n_cards:
                break
            h = row_heights[r]
            draw_card(x_cursor, y_cursor, card_width, h, card_texts[idx])
            x_cursor += card_width + gap
            idx += 1
        y_cursor += row_heights[r] + gap

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return out_path


# -------------------------
# Main bundle builder
# -------------------------

def make_visual_bundle(
    result: Dict[str, Any],
    y: str,
    output_dir: str = "/workspaces/Kairos/outputs",
    run_name: Optional[str] = None,
    render_formats: Optional[List[str]] = None,  # e.g. ["svg", "png", "pdf"]
) -> Dict[str, Any]:
    """
    Produces DOT and optionally rendered files for:
      - full dag
      - abstract dag (explanatory projection)
      - chain cards PNG (no Graphviz)

    Saves:
      - <run_name>_full.dot
      - <run_name>_abstract.dot
      - <run_name>_full.{svg/png/pdf}  (if Graphviz available)
      - <run_name>_abstract.{svg/png/pdf} (if Graphviz available)
      - <run_name>_chain_cards.png
    """
    if render_formats is None:
        render_formats = ["png"]  # keep default simple

    os.makedirs(output_dir, exist_ok=True)

    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"kairos_{ts}"

    dag: nx.DiGraph = result["dag"]
    effects = result.get("effects", []) or []
    selected = result.get("selected_drivers", []) or []

    # DOT strings
    full_dot = dag_to_dot_styled(
        dag=dag,
        y=y,
        effects=effects,
        selected_drivers=selected,
        title="Kairos — Full Causal Graph",
        show_edge_sources=True,
    )

    abstract_dag = build_explanatory_abstract_dag(
        dag=dag,
        y=y,
        selected_drivers=selected,
    )

    abstract_dot = dag_to_dot_styled(
        dag=abstract_dag,
        y=y,
        effects=effects,
        selected_drivers=selected,
        title="Kairos — Abstract Causal Graph (Explanatory Projection)",
        show_edge_sources=True,
    )

    # Save DOT files
    full_dot_path = os.path.join(output_dir, f"{run_name}_full.dot")
    abstract_dot_path = os.path.join(output_dir, f"{run_name}_abstract.dot")
    with open(full_dot_path, "w", encoding="utf-8") as f:
        f.write(full_dot)
    with open(abstract_dot_path, "w", encoding="utf-8") as f:
        f.write(abstract_dot)

    bundle: Dict[str, Any] = {
        "run_name": run_name,
        "output_dir": output_dir,
        "paths": {
            "full_dot": full_dot_path,
            "abstract_dot": abstract_dot_path,
        },
        "full_dot": full_dot,
        "abstract_dot": abstract_dot,
        "tables": {
            "effects": effects_table(result),
            "selected": selected_drivers_table(result),
            "counterfactuals": counterfactuals_table(result),
            "eliminated": eliminated_table(result),
        },
    }

    # Causal chains (prefer longer when available)
    bundle["causal_chains"] = extract_causal_chains(
        dag=dag,
        y=y,
        selected_drivers=selected,
        max_chain_length=4,
        max_chains_per_driver=3,
    )

    # Chain cards PNG (no Graphviz required)
    try:
        cards_path = os.path.join(output_dir, f"{run_name}_chain_cards.png")
        bundle["paths"]["chain_cards_png"] = render_chain_cards_png(
            y=y,
            selected_drivers=selected,
            causal_chains=bundle.get("causal_chains", {}),
            counterfactuals=result.get("counterfactuals", {}),
            baseline=result.get("baseline", {}),
            out_path=cards_path,
        )
    except Exception as e:
        bundle.setdefault("render_errors", {})
        bundle["render_errors"]["chain_cards_png"] = str(e)

    # Optional Graphviz rendering (may fail if dot is not installed)
    render_errors: Dict[str, str] = {}
    for fmt in render_formats:
        try:
            full_base = os.path.join(output_dir, f"{run_name}_full")
            abs_base = os.path.join(output_dir, f"{run_name}_abstract")

            full_rendered = render_dot_to_file(full_dot, out_path=full_base, fmt=fmt)
            abs_rendered = render_dot_to_file(abstract_dot, out_path=abs_base, fmt=fmt)

            bundle["paths"][f"full_{fmt}"] = full_rendered
            bundle["paths"][f"abstract_{fmt}"] = abs_rendered

            if fmt.lower() == "svg":
                try:
                    with open(full_rendered, "r", encoding="utf-8") as f:
                        bundle["full_svg"] = f.read()
                    with open(abs_rendered, "r", encoding="utf-8") as f:
                        bundle["abstract_svg"] = f.read()
                except Exception as e:
                    render_errors["svg_readback"] = str(e)

        except Exception as e:
            render_errors[str(fmt)] = str(e)

    if render_errors:
        bundle.setdefault("render_errors", {})
        bundle["render_errors"].update(render_errors)

    return bundle
def render_nx_fallback_png(G, out_path: str, *, figsize=(14, 10)) -> str:
    """
    Pure-Python fallback PNG renderer using networkx + matplotlib.
    Works even when Graphviz 'dot' executable is not installed.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=figsize)
    try:
        pos = nx.spring_layout(G, seed=42, k=0.8)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", width=1.2)
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path
