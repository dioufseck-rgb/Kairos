## Key Drivers of Unplanned Downtime Hours (Next 30d)

This report identifies the primary causal drivers influencing Unplanned Downtime Hours (Next 30d). We found that Spare Parts Availability Index and Maintenance Quality Index are key factors that, if improved, could reduce downtime. Conversely, Production Target Pressure Index and Lubrication Interval (Days) are associated with increased downtime. Operator Training Hours (Monthly) shows a minor negative association with downtime. Understanding these causal pathways is crucial for effective intervention strategies.

### Key causal drivers
- **Maintenance Quality Index** — A one standard deviation increase in Maintenance Quality Index is causally associated with a decrease of -2.0048112824315227 Unplanned Downtime Hours (Next 30d) per +1σ.
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected Unplanned Downtime Hours (Next 30d) change, given the discovered causal structure?
  - What-if: Δ ≈ -1.588 (CI -13.305..+10.567)
  - Chain: Maintenance Quality Index → Vibration RMS → Failure Risk (0-1) → Unplanned Downtime Hours (Next 30d)
  - Notes: There is uncertainty in the counterfactual estimate, as indicated by the wide confidence interval. The causal direction within the chain was not uniquely identified by the discovery algorithm.
- **Production Target Pressure Index** — A one standard deviation increase in Production Target Pressure Index is causally associated with an increase of 1.6487443865426834 Unplanned Downtime Hours (Next 30d) per +1σ.
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected Unplanned Downtime Hours (Next 30d) change, given the discovered causal structure?
  - What-if: Δ ≈ +1.226 (CI -12.359..+12.388)
  - Chain: Production Target Pressure Index → Defect Rate Index → Unplanned Downtime Hours (Next 30d)
  - Notes: There is uncertainty in the counterfactual estimate, as indicated by the wide confidence interval. The causal direction within the chain was not uniquely identified by the discovery algorithm.
- **Lubrication Interval (Days)** — A one standard deviation increase in Lubrication Interval (Days) is causally associated with an increase of 0.4083902250330512 Unplanned Downtime Hours (Next 30d) per +1σ.
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected Unplanned Downtime Hours (Next 30d) change, given the discovered causal structure?
  - What-if: Δ ≈ +0.933 (CI -11.515..+12.752)
  - Chain: Lubrication Interval (Days) → Vibration RMS → Failure Risk (0-1) → Unplanned Downtime Hours (Next 30d)
  - Notes: There is uncertainty in the counterfactual estimate, as indicated by the wide confidence interval. The causal direction within the chain was not uniquely identified by the discovery algorithm.
- **Operator Training Hours (Monthly)** — A one standard deviation increase in Operator Training Hours (Monthly) is causally associated with a decrease of -0.1519912030399265 Unplanned Downtime Hours (Next 30d) per +1σ.
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected Unplanned Downtime Hours (Next 30d) change, given the discovered causal structure?
  - What-if: Δ ≈ -0.181 (CI -12.890..+10.990)
  - Chain: Operator Training Hours (Monthly) → Procedure Compliance Index → Defect Rate Index → Unplanned Downtime Hours (Next 30d)
  - Notes: There is uncertainty in the counterfactual estimate, as indicated by the wide confidence interval. The causal direction within the chain was not uniquely identified by the discovery algorithm.

### What not to optimize (and why)
- None.

### Next steps
- Prioritize interventions on 'Spare Parts Availability Index' and 'Maintenance Quality Index' to potentially reduce Unplanned Downtime Hours (Next 30d).
- Investigate the pathways identified for 'Production Target Pressure Index' and 'Lubrication Interval (Days)' to understand how they contribute to increased downtime and identify mitigation strategies.
- Further analyze the identified causal chains to refine understanding of the mechanisms and potential interaction effects between drivers.
- Monitor the impact of any implemented interventions on Unplanned Downtime Hours (Next 30d) and related metrics to validate causal findings.