# =========================
# Visualization helpers (DOT + SVG + abstraction graph)
# =========================
from typing import List, Dict, Any, Optional, Iterable
import networkx as nx
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import textwrap
from PIL import Image, ImageDraw, ImageFont



def build_explanatory_abstract_dag(
    dag: nx.DiGraph,
    y: str,
    selected_drivers: List[Dict[str, Any]],
) -> nx.DiGraph:
    """
    Build an explanatory (Strevens-style) abstract graph.

    Semantics:
    - Every selected driver must explain Y
    - Explanation may be:
        * via an inferred causal path in the discovered DAG, or
        * via a difference-making causal effect (ATE-based)
    """
    G = nx.DiGraph()
    G.add_node(y)

    for d in selected_drivers:
        x = d.get("x")
        if x is None:
            continue

        G.add_node(x)

        # Check if any path exists in the discovered graph
        has_path = False
        try:
            has_path = nx.has_path(dag.to_undirected(), x, y)
        except Exception:
            has_path = False

        if has_path:
            label = "inferred causal path"
        else:
            label = "difference-making (effect-based)"

        G.add_edge(x, y, source=label)

    return G

def extract_causal_chains(
    dag: nx.DiGraph,
    y: str,
    selected_drivers: List[Dict[str, Any]],
    max_chain_length: int = 4,
    max_chains_per_driver: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract causal chains from selected drivers to y.

    Returns:
      {
        driver_x: [
            {
              "chain": [x, ..., y],
              "type": "discovered" | "abstracted",
              "length": int
            },
            ...
        ]
      }
    """
    chains = {}
    UG = dag.to_undirected()

    for d in selected_drivers:
        x = d.get("x")
        if x is None:
            continue

        driver_chains = []

        # Try to extract actual paths from discovery graph
        try:
            all_paths = nx.all_simple_paths(
                UG, source=x, target=y, cutoff=max_chain_length
            )
            for p in all_paths:
                if len(driver_chains) >= max_chains_per_driver:
                    break
                driver_chains.append({
                    "chain": p,
                    "type": "discovered",
                    "length": len(p) - 1
                })
        except Exception:
            pass

        # If no discovered chain exists, add abstract chain
        if not driver_chains:
            driver_chains.append({
                "chain": [x, y],
                "type": "abstracted (difference-making)",
                "length": 1
            })

        chains[x] = driver_chains

    return chains


def _escape_dot(s: str) -> str:
    return s.replace('"', '\\"')

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def build_abstract_dag(
    dag: nx.DiGraph,
    y: str,
    drivers: List[Dict[str, Any]],
    include_connectors: bool = True,
    max_connector_hops: int = 4,
) -> nx.DiGraph:
    """
    Build an abstract graph focusing on y + selected drivers.
    Optionally include minimal connector nodes along shortest paths driver -> y (or y -> driver if direction differs).
    """
    driver_nodes = [d["x"] for d in drivers if "x" in d]
    keep = set([y] + driver_nodes)

    if include_connectors:
        # Use undirected shortest paths to get connectors, but keep original directions
        UG = dag.to_undirected()
        for x in driver_nodes:
            if x not in dag.nodes or y not in dag.nodes:
                continue
            try:
                path = nx.shortest_path(UG, source=x, target=y)
                if len(path) - 1 <= max_connector_hops:
                    keep.update(path)
            except Exception:
                pass

    sub = dag.subgraph(sorted(keep)).copy()
    return sub

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
    eff_map = {}
    if effects:
        for e in effects:
            if "x" in e and "effect" in e and "error" not in e:
                eff_map[e["x"]] = _safe_float(e["effect"], 0.0)

    selected = set()
    if selected_drivers:
        selected = {d.get("x") for d in selected_drivers if d.get("x")}

    # scale width by |effect|
    abs_vals = [abs(v) for v in eff_map.values() if v is not None]
    max_abs = max(abs_vals) if abs_vals else 0.0

    def width_for(node: str) -> float:
        if node not in eff_map or max_abs <= 0:
            return 1.2
        # 1.2 .. 4.0
        return 1.2 + 2.8 * (abs(eff_map[node]) / max_abs)

    def node_fill(node: str) -> str:
        if node == y:
            return "#FDE68A"  # warm highlight
        if node in selected:
            return "#DBEAFE"  # light blue
        return "#F9FAFB"     # light gray

    def node_border(node: str) -> float:
        if node == y:
            return 3.0
        if node in selected:
            return 2.2
        return 1.2

    lines = []
    lines.append("digraph Kairos {")
    lines.append('  rankdir="LR";')
    lines.append('  graph [fontname="Helvetica", fontsize=18, labelloc="t"];')
    lines.append('  node  [fontname="Helvetica", fontsize=12, shape=box, style="rounded,filled"];')
    lines.append('  edge  [fontname="Helvetica", fontsize=10, arrowsize=0.8];')
    lines.append(f'  label="{_escape_dot(title)}";')

    # nodes
    for n in dag.nodes():
        fill = node_fill(n)
        penw = node_border(n)
        w = width_for(n)
        # label includes effect if present
        if n in eff_map:
            label = f"{n}\\nATE={eff_map[n]:.3f}"
        else:
            label = n
        lines.append(
            f'  "{_escape_dot(n)}" [label="{_escape_dot(label)}", fillcolor="{fill}", penwidth={penw:.2f}, width={w:.2f}];'
        )

    # edges
    for u, v, d in dag.edges(data=True):
        src = d.get("source", "")
        if show_edge_sources and src:
            elabel = src
        else:
            elabel = ""

        # highlight edges into outcome and between selected nodes
        penw = 1.2
        color = "#6B7280"  # gray
        if v == y:
            penw = 2.2
            color = "#111827"  # near-black
        if u in selected and v in selected:
            penw = max(penw, 2.0)
            color = "#1D4ED8"  # blue

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
    Render DOT to SVG string. Requires python 'graphviz' package + system graphviz installed.
    """
    try:
        import graphviz
    except Exception as e:
        raise ImportError("Missing dependency: graphviz. Add 'graphviz' to requirements.txt") from e

    src = graphviz.Source(dot)
    # graphviz.Source.pipe returns bytes
    svg_bytes = src.pipe(format="svg")
    return svg_bytes.decode("utf-8")

def render_dot_to_file(dot: str, out_path: str, fmt: str = "png") -> str:
    """
    Write DOT render to a file (png/svg/pdf). Returns file path.
    Requires system graphviz.
    """
    try:
        import graphviz
    except Exception as e:
        raise ImportError("Missing dependency: graphviz. Add 'graphviz' to requirements.txt") from e

    src = graphviz.Source(dot)
    rendered = src.render(filename=out_path, format=fmt, cleanup=True)
    return rendered

def effects_table(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for e in result.get("effects", []):
        rows.append({
            "treatment": e.get("x"),
            "ATE": e.get("effect"),
            "rows_used": e.get("rows_used"),
            "confounders_used": ", ".join(e.get("confounders_used", []) or []),
            "method": e.get("method"),
            "error": e.get("error"),
        })
    return pd.DataFrame(rows).sort_values(by=["ATE"], key=lambda s: s.abs(), ascending=False, na_position="last")

def selected_drivers_table(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for d in result.get("selected_drivers", []):
        rows.append({
            "driver": d.get("x"),
            "ATE": d.get("effect"),
            "rows_used": d.get("rows_used"),
            "confounders_used": ", ".join(d.get("confounders_used", []) or []),
        })
    return pd.DataFrame(rows).sort_values(by=["ATE"], key=lambda s: s.abs(), ascending=False, na_position="last")

def counterfactuals_table(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    cf = result.get("counterfactuals", {}) or {}
    for k, v in cf.items():
        if isinstance(v, dict) and "error" in v:
            rows.append({"driver": k, "cf_mean": None, "ci_low": None, "ci_high": None, "error": v["error"]})
        else:
            ci = (v or {}).get("counterfactual_ci", [None, None])
            rows.append({
                "driver": k,
                "cf_mean": (v or {}).get("counterfactual_mean"),
                "ci_low": ci[0] if len(ci) > 0 else None,
                "ci_high": ci[1] if len(ci) > 1 else None,
                "error": None
            })
    return pd.DataFrame(rows)

def eliminated_table(result: Dict[str, Any]) -> pd.DataFrame:
    elim = result.get("eliminated", {}) or {}
    rows = [{"variable": k, "reason": v} for k, v in elim.items()]
    return pd.DataFrame(rows)

def _default_font(size: int = 18) -> ImageFont.FreeTypeFont:
    """
    Try to use a TTF font if available; fall back to PIL default.
    """
    try:
        # Common fonts in many Linux images
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _wrap_lines(text: str, width_chars: int) -> List[str]:
    lines = []
    for para in text.split("\n"):
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
      - Counterfactual summary if available
    """
    if counterfactuals is None:
        counterfactuals = {}

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
            top_chain = chains[0]
            chain_path = " → ".join(top_chain["chain"])
            chain_type = top_chain.get("type", "unknown")
        else:
            chain_path = f"{x} → {y}"
            chain_type = "abstracted"

        cf = counterfactuals.get(x)
        if isinstance(cf, dict) and cf and "error" not in cf and "counterfactual_mean" in cf:
            cf_mean = cf.get("counterfactual_mean")
            ci = cf.get("counterfactual_ci", [None, None])
            cf_text = f"What-if: do({x}) ⇒ {y} ≈ {cf_mean:.3f} (CI {ci[0]:.3f}..{ci[1]:.3f})"
        elif isinstance(cf, dict) and cf and "error" in cf:
            cf_text = f"What-if: (not available) {cf['error']}"
        else:
            cf_text = "What-if: (not available)"

        txt = (
            f"{x}\n"
            f"ATE: {ate_str}\n\n"
            f"Chain ({chain_type}):\n"
            f"{chain_path}\n\n"
            f"{cf_text}"
        )
        card_texts.append(txt)

    if not card_texts:
        # Render a simple image saying nothing selected
        img = Image.new("RGB", (900, 180), "white")
        draw = ImageDraw.Draw(img)
        draw.text((24, 24), f"No selected drivers to display for {y}.", fill="black", font=title_font)
        img.save(out_path)
        return out_path

    # Measure card heights (approx) based on wrapped lines
    def measure_card_height(text: str) -> int:
        lines = _wrap_lines(text, wrap_chars)
        # line heights
        h_title = 28
        h_body = 22
        height = padding * 2
        for i, ln in enumerate(lines):
            # crude heuristic: first line is "title"
            height += h_title if i == 0 else (h_body if ln.strip() else 10)
        return max(card_min_height, height)

    card_heights = [measure_card_height(t) for t in card_texts]
    n_cards = len(card_texts)
    rows = (n_cards + cards_per_row - 1) // cards_per_row

    # Compute row heights
    row_heights = []
    for r in range(rows):
        start = r * cards_per_row
        end = min(start + cards_per_row, n_cards)
        row_heights.append(max(card_heights[start:end]))

    img_width = cards_per_row * card_width + (cards_per_row - 1) * gap + 2 * padding
    img_height = sum(row_heights) + (rows - 1) * gap + 2 * padding + 54  # top banner

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # Header
    header = f"Kairos — Causal Chain Cards (Outcome: {y})"
    draw.text((padding, padding), header, fill="black", font=title_font)
    y_cursor = padding + 44

    # Card drawing helpers
    def draw_card(x0: int, y0: int, w: int, h: int, text: str):
        # card background
        draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=18, fill="#F9FAFB", outline="#111827", width=3)

        # text
        lines = _wrap_lines(text, wrap_chars)
        tx = x0 + padding
        ty = y0 + padding

        for i, ln in enumerate(lines):
            if i == 0:
                # driver title line
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

    # Draw cards grid
    idx = 0
    for r in range(rows):
        x_cursor = padding
        for c in range(cards_per_row):
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


def make_visual_bundle(
    result: Dict[str, Any],
    y: str,
    output_dir: str = "/workspaces/Kairos/outputs",
    run_name: Optional[str] = None,
    render_formats: Optional[List[str, ]] = ["png"],  # e.g. ["svg", "png", "pdf"]
) -> Dict[str, Any]:
    """
    Produce DOT + (optional) rendered files for:
      - full dag
      - abstract dag (y + selected drivers + connectors)

    Saves:
      - <run_name>_full.dot
      - <run_name>_abstract.dot
      - <run_name>_full.{svg/png/pdf}
      - <run_name>_abstract.{svg/png/pdf}
    """
    if render_formats is None:
        render_formats = ["svg"]  # default

    os.makedirs(output_dir, exist_ok=True)

    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"kairos_{ts}"

    dag = result["dag"]
    effects = result.get("effects", [])
    selected = result.get("selected_drivers", [])

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
    selected_drivers=selected,)

    abstract_dot = dag_to_dot_styled(
        dag=abstract_dag,
        y=y,
        effects=effects,
        selected_drivers=selected,
        title="Kairos — Abstract Causal Graph (Drivers + Connectors)",
        show_edge_sources=True,
    )

    # --- Save DOT files
    full_dot_path = os.path.join(output_dir, f"{run_name}_full.dot")
    abstract_dot_path = os.path.join(output_dir, f"{run_name}_abstract.dot")
    with open(full_dot_path, "w", encoding="utf-8") as f:
        f.write(full_dot)
    with open(abstract_dot_path, "w", encoding="utf-8") as f:
        f.write(abstract_dot)

    bundle = {
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

    # --- Try rendering to files (Graphviz required)
    render_errors = {}
    for fmt in render_formats:
        try:
            # graphviz.Source.render expects a filename *without* extension; it adds it.
            full_base = os.path.join(output_dir, f"{run_name}_full")
            abs_base = os.path.join(output_dir, f"{run_name}_abstract")

            full_rendered = render_dot_to_file(full_dot, out_path=full_base, fmt=fmt)
            abs_rendered = render_dot_to_file(abstract_dot, out_path=abs_base, fmt=fmt)

            bundle["paths"][f"full_{fmt}"] = full_rendered
            bundle["paths"][f"abstract_{fmt}"] = abs_rendered

            # If svg, also store in-memory SVG strings for notebook/UI embedding
            if fmt.lower() == "svg":
                try:
                    with open(full_rendered, "r", encoding="utf-8") as f:
                        bundle["full_svg"] = f.read()
                    with open(abs_rendered, "r", encoding="utf-8") as f:
                        bundle["abstract_svg"] = f.read()
                except Exception as e:
                    render_errors["svg_readback"] = str(e)

        except Exception as e:
            render_errors[fmt] = str(e)

    if render_errors:
        bundle["render_errors"] = render_errors
    

        # --- Causal chains
    bundle["causal_chains"] = extract_causal_chains(
        dag=dag,
        y=y,
        selected_drivers=selected,
        max_chain_length=4,
        max_chains_per_driver=3,
    )

    # --- Chain cards PNG (no Graphviz needed)
    try:
        cards_path = os.path.join(output_dir, f"{run_name}_chain_cards.png")
        bundle["paths"]["chain_cards_png"] = render_chain_cards_png(
            y=y,
            selected_drivers=selected,
            causal_chains=bundle.get("causal_chains", {}),
            counterfactuals=result.get("counterfactuals", {}),
            out_path=cards_path,
        )
    except Exception as e:
        bundle.setdefault("render_errors", {})
        bundle["render_errors"]["chain_cards_png"] = str(e)

    return bundle
