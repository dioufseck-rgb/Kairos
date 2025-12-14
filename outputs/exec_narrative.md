## Key Drivers of Unplanned Downtime Hours (Next 30d)

Our analysis identifies several key drivers influencing Unplanned Downtime Hours (Next 30d). Improving Spare Parts Availability Index and Maintenance Quality Index are associated with reductions in downtime, while increases in Production Target Pressure Index and Lubrication Interval (Days) are linked to increased downtime. Operator Training Hours (Monthly) shows a minor association with reduced downtime. Understanding these pathways is crucial for targeted interventions, though the causal direction for these pathways is not uniquely identified, and counterfactual predictions carry significant uncertainty.

### Key causal drivers
- **Spare Parts Availability Index** — A one standard deviation increase in Spare Parts Availability Index is associated with a decrease of -2.108245859802473 Unplanned Downtime Hours (Next 30d).
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected Unplanned Downtime Hours (Next 30d) change, given the discovered causal structure?
  - What-if: Δ ≈ -2.061 (CI -13.165..+8.504)
  - Chain: Spare Parts Availability Index → Repair Duration (Hours) → Unplanned Downtime Hours (Next 30d)
  - Notes: There is significant uncertainty in the counterfactual prediction, as indicated by the wide confidence interval. Additionally, the causal direction for this pathway was not uniquely identified from the data.
- **Maintenance Quality Index** — A one standard deviation increase in Maintenance Quality Index is associated with a decrease of -2.0048112824315227 Unplanned Downtime Hours (Next 30d).
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected Unplanned Downtime Hours (Next 30d) change, given the discovered causal structure?
  - What-if: Δ ≈ -1.732 (CI -13.779..+9.721)
  - Chain: Maintenance Quality Index → Vibration RMS → Failure Risk (0-1) → Unplanned Downtime Hours (Next 30d)
  - Notes: There is significant uncertainty in the counterfactual prediction, as indicated by the wide confidence interval. Additionally, the causal direction for this pathway was not uniquely identified from the data.
- **Production Target Pressure Index** — A one standard deviation increase in Production Target Pressure Index is associated with an increase of 1.6487443865426834 Unplanned Downtime Hours (Next 30d).
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected Unplanned Downtime Hours (Next 30d) change, given the discovered causal structure?
  - What-if: Δ ≈ +1.163 (CI -10.717..+12.659)
  - Chain: Production Target Pressure Index → Defect Rate Index → Unplanned Downtime Hours (Next 30d)
  - Notes: There is significant uncertainty in the counterfactual prediction, as indicated by the wide confidence interval. Additionally, the causal direction for this pathway was not uniquely identified from the data.
- **Lubrication Interval (Days)** — A one standard deviation increase in Lubrication Interval (Days) is associated with an increase of 0.4083902250330512 Unplanned Downtime Hours (Next 30d).
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected Unplanned Downtime Hours (Next 30d) change, given the discovered causal structure?
  - What-if: Δ ≈ +1.016 (CI -10.755..+12.420)
  - Chain: Lubrication Interval (Days) → Vibration RMS → Failure Risk (0-1) → Unplanned Downtime Hours (Next 30d)
  - Notes: There is significant uncertainty in the counterfactual prediction, as indicated by the wide confidence interval. Additionally, the causal direction for this pathway was not uniquely identified from the data.
- **Operator Training Hours (Monthly)** — A one standard deviation increase in Operator Training Hours (Monthly) is associated with a decrease of -0.1519912030399265 Unplanned Downtime Hours (Next 30d).
  - If we intervene on this driver in a realistic way (p50 → p75 (median to 75th percentile)), how would expected Unplanned Downtime Hours (Next 30d) change, given the discovered causal structure?
  - What-if: Δ ≈ -0.154 (CI -13.523..+11.722)
  - Chain: Operator Training Hours (Monthly) → Procedure Compliance Index → Defect Rate Index → Unplanned Downtime Hours (Next 30d)
  - Notes: There is significant uncertainty in the counterfactual prediction, as indicated by the wide confidence interval. Additionally, the causal direction for this pathway was not uniquely identified from the data.

### What not to optimize (and why)
- None.

### Next steps
- Prioritize interventions on 'Spare Parts Availability Index' and 'Maintenance Quality Index' to potentially reduce Unplanned Downtime Hours, acknowledging the uncertainty in predictions.
- Investigate the relationship between 'Production Target Pressure Index' and 'Lubrication Interval (Days)' with Unplanned Downtime Hours to understand if current operational targets or practices are inadvertently increasing downtime.
- Conduct further analysis or pilot programs to validate the identified causal pathways and refine counterfactual predictions, especially given the wide confidence intervals and undirected nature of the causal chains.