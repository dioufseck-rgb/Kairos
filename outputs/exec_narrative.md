## Causal Analysis of Final Arrival Delay

This report identifies key drivers influencing 'final_arrival_delay_min'. Our analysis, based on 5000 rows of data, reveals that 'dwell_passenger_delay_s2' and 'dwell_passenger_delay_s5' are the most significant causal factors, with 'connection_hold_delay_s2', 'track_delay_g5', and 'track_delay_g2' also contributing to delays. Interventions targeting passenger dwell delays show potential for impact, though with considerable uncertainty in the estimated effect range. The causal pathways identified are based on undirected shortest paths, indicating that while a relationship exists, the precise directionality of all links within the chain was not uniquely identified.

### Key causal drivers
- **connection_hold_delay_s2** — For every +1 standard deviation increase in connection_hold_delay_s2, final_arrival_delay_min is expected to increase by 1.316 minutes, holding all other factors constant.
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected final_arrival_delay_min change, given the discovered causal structure?
  - What-if: counterfactual not available (SCM skipped: non-numeric / low-variance / discrete intervention variable.)
  - Chain: connection_hold_delay_s2 → arrival_delay_s2 → arrival_delay_s3 → arrival_delay_s4 → arrival_delay_s5 → final_arrival_delay_min
  - Notes: The causal chain was identified as an undirected shortest path, meaning the direction of causality for all links within the chain was not uniquely identified. Counterfactual analysis was skipped due to the intervention variable's characteristics.
- **track_delay_g5** — For every +1 standard deviation increase in track_delay_g5, final_arrival_delay_min is expected to increase by 1.057 minutes, holding all other factors constant.
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected final_arrival_delay_min change, given the discovered causal structure?
  - What-if: counterfactual not available (No (directed or salvageable) causal path to outcome in discovered graph; SCM what-if skipped.)
  - Chain: track_delay_g5 → segment_total_delay_g5 → arrival_delay_s1 → arrival_delay_s2 → arrival_delay_s3 → final_arrival_delay_min
  - Notes: The causal chain was identified as an undirected shortest path, meaning the direction of causality for all links within the chain was not uniquely identified. Counterfactual analysis was skipped because no directed or salvageable causal path to the outcome was found in the discovered graph.
- **track_delay_g2** — For every +1 standard deviation increase in track_delay_g2, final_arrival_delay_min is expected to increase by 1.054 minutes, holding all other factors constant.
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected final_arrival_delay_min change, given the discovered causal structure?
  - What-if: counterfactual not available (No (directed or salvageable) causal path to outcome in discovered graph; SCM what-if skipped.)
  - Chain: track_delay_g2 → segment_total_delay_g2 → arrival_delay_s1 → arrival_delay_s2 → arrival_delay_s3 → final_arrival_delay_min
  - Notes: The causal chain was identified as an undirected shortest path, meaning the direction of causality for all links within the chain was not uniquely identified. Counterfactual analysis was skipped because no directed or salvageable causal path to the outcome was found in the discovered graph.
- **dwell_passenger_delay_s2** — For every +1 standard deviation increase in dwell_passenger_delay_s2, final_arrival_delay_min is expected to increase by 2.250 minutes, holding all other factors constant.
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected final_arrival_delay_min change, given the discovered causal structure?
  - What-if: Δ ≈ +0.918 (CI -18.465..+24.116)
  - Chain: dwell_passenger_delay_s2 → passengers → dwell_passenger_delay_s6 → final_arrival_delay_min
  - Notes: The causal chain was identified as an undirected shortest path, meaning the direction of causality for all links within the chain was not uniquely identified. The counterfactual estimate shows a wide confidence interval, indicating significant uncertainty in the precise impact of this intervention.
- **dwell_passenger_delay_s5** — For every +1 standard deviation increase in dwell_passenger_delay_s5, final_arrival_delay_min is expected to increase by 2.197 minutes, holding all other factors constant.
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected final_arrival_delay_min change, given the discovered causal structure?
  - What-if: Δ ≈ +1.155 (CI -18.425..+26.900)
  - Chain: dwell_passenger_delay_s5 → passengers → dwell_passenger_delay_s6 → final_arrival_delay_min
  - Notes: The causal chain was identified as an undirected shortest path, meaning the direction of causality for all links within the chain was not uniquely identified. The counterfactual estimate shows a wide confidence interval, indicating significant uncertainty in the precise impact of this intervention.

### What not to optimize (and why)
- None.

### Next steps
- Validate the identified causal pathways and their directionality with domain experts to refine the understanding of these complex systems.
- Investigate the specific mechanisms through which 'dwell_passenger_delay_s2' and 'dwell_passenger_delay_s5' influence 'passengers' and subsequently 'dwell_passenger_delay_s6' to better target interventions.
- Prioritize interventions on drivers with available counterfactual estimates, while acknowledging the wide confidence intervals and inherent uncertainty.
- Collect additional data or conduct targeted experiments to reduce uncertainty in counterfactual estimates and confirm the causal directions for 'undirected_shortest' paths.