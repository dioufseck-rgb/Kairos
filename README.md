# Kairos

Kairos is a small toolkit for discovery-driven causal analysis of tabular operational datasets.

It combines constraint-based causal discovery (PC / FCI), structural causal models (DoWhy),
effect estimation (EconML), and lightweight visualization & narrative scaffolding to help
teams find plausible, actionable difference-makers in predictive-maintenance and manufacturing style datasets.

Key features
- Causal discovery with PC and FCI (via `causal-learn`)
- Structural causal models and counterfactual sampling (DoWhy / GCM)
- Heterogeneous effect estimation with `econml` (ATE / DML)
- Visual exports (DOT / PNG) and convenience plotting (`kairos_viz.py`)
- Optional executive narratives via LLM provider (Gemini; opt-in via YAML)

Quick start

1. Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare or generate example data:

```bash
# Build the C-MAPSS multisnapshot CSV used by the included example
python prep_cmapss.py

# Or generate synthetic datasets
python make_synth_data.py
python synth_manufacturing.py
```

3. Run the Kairos core pipeline with a YAML spec (examples provided):

```bash
python kairos_core.py kairos_run_cmapss_fd001.yaml
python kairos_core.py kairos_run_manufacturing.yaml
```

Outputs
- Default outputs are written to `outputs/` (configurable via the YAML spec). Typical files include:
	- `*_full.dot`, `*_abstract.dot` — Graphviz DOT exports
	- `*_bundle.json` — structured results bundle
	- `exec_narrative.md` — LLM-generated narrative (if enabled)
	- PNGs / rendered images when you run the visualization helpers

Visualization
- Quick demo rendering: `python draw_graph.py` writes `kairos_causal_graph.png`.
- For rich, styled DOT renders use `kairos_viz.py` helpers and the `graphviz` system package (dot). The Python package `graphviz` is also required for programmatic renders.

Configuration
- Example run specifications live in `kairos_run_cmapss_fd001.yaml` and `kairos_run_manufacturing.yaml`.
- The YAML controls dataset location, outcome column, variable roles (controllable / uncontrollable), preprocessing, algorithm hyperparameters, output directory, and optional LLM settings.

LLM/Executive Narratives
- Enable narratives in the YAML by setting `llm.enabled: true` and configuring `provider`/`model` and `api_key_env`.
- For Gemini, set the API key in the environment variable named in `api_key_env` (example: `export GEMINI_KEY=...`).

Development notes
- The core pipeline is implemented in `kairos_core.py` (entry: `python kairos_core.py /path/to/spec.yaml`).
- Visualization helpers are in `kairos_viz.py` and `draw_graph.py` (small demo layout).
- Data preprocessing utilities are in `prep_cmapss.py` and the synthetic generators are `make_synth_data.py` and `synth_manufacturing.py`.

Dependencies
- See `requirements.txt`. Some notable components: `pandas`, `numpy`, `networkx`, `causal-learn`, `dowhy`, `econml`, `scikit-learn`, and `graphviz` (system package + Python binding for rendering).

Caveats & guidance
- Constraint-based discovery (PC / FCI) assumes reasonably sized datasets and numeric inputs; the repo includes helpers to coerce and filter columns prior to discovery.
- Always validate causal claims with domain knowledge before acting — these are discovery-and-estimate tools, not causal proof.

Contributing
- Bug reports and PRs welcome. Add tests or examples that clarify expected behavior.

License
- No license file is included in this repository. If you plan to reuse or distribute this code, please add an explicit license.

Contact
- File issues or PRs on the repository to discuss improvements or report problems.

Enjoy causal tinkering!