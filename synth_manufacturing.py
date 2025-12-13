import os
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_synth_manufacturing(
    n: int = 6000,
    seed: int = 42,
    missing_rate: float = 0.03,
    out_csv: str = "/workspaces/Kairos/synth_manufacturing.csv",
):
    """
    Synthetic dataset: manufacturing reliability + quality + throughput.
    Outcome is continuous: 'Unplanned Downtime Hours (Next 30d)' (lower is better).

    Key causal chains (multi-hop):
      Maintenance Quality -> Vibration RMS -> Failure Risk -> Downtime
      Operator Training -> Procedure Compliance -> Defect Rate -> Downtime
      Spare Parts Availability -> Repair Duration -> Downtime
      Shift -> Fatigue -> Procedure Compliance -> Defect Rate -> Downtime
      Ambient Temperature -> Cooling Efficiency -> Vibration RMS -> Failure Risk -> Downtime
      Machine Age -> Wear Index -> Vibration RMS -> Failure Risk -> Downtime

    Confounding examples:
      Machine Age affects Wear & Vibration & Failure Risk & Downtime.
      Shift affects Fatigue and indirectly impacts several measures.
    """
    rng = np.random.default_rng(seed)

    # Uncontrollable / context (some categorical)
    plant = rng.choice(["Plant A", "Plant B", "Plant C"], size=n, p=[0.4, 0.35, 0.25])
    machine_type = rng.choice(["CNC", "Press", "Conveyor"], size=n, p=[0.45, 0.35, 0.20])
    shift = rng.choice(["Day", "Swing", "Night"], size=n, p=[0.45, 0.35, 0.20])

    ambient_temp_c = rng.normal(24, 5, size=n) + (shift == "Night") * 1.0
    humidity_pct = np.clip(rng.normal(55, 12, size=n), 20, 95)

    machine_age_years = np.clip(rng.normal(8, 3, size=n) + (machine_type == "Press") * 1.0, 1, 20)

    # Controllable levers (numeric, actionable)
    maintenance_quality = np.clip(rng.normal(0.0, 1.0, size=n), -2.5, 2.5)  # higher is better
    lubrication_interval_days = np.clip(rng.normal(14, 4, size=n), 3, 30)    # lower is better
    operator_training_hours = np.clip(rng.normal(6, 3, size=n), 0, 20)       # higher is better
    spare_parts_availability = np.clip(rng.normal(0.0, 1.0, size=n), -2.5, 2.5)  # higher is better
    production_target_pressure = np.clip(rng.normal(0.0, 1.0, size=n), -2.5, 2.5)  # higher increases risk

    # Mediators / internal states
    fatigue = (
        0.9 * (shift == "Night").astype(float)
        + 0.4 * (shift == "Swing").astype(float)
        + 0.15 * (ambient_temp_c - 24) / 5
        + rng.normal(0, 0.4, size=n)
    )

    procedure_compliance = (
        0.7 * (operator_training_hours / 10.0)
        - 0.6 * fatigue
        + rng.normal(0, 0.5, size=n)
    )

    wear_index = (
        0.35 * machine_age_years
        + 0.25 * (production_target_pressure + 0.2 * (machine_type == "Press").astype(float))
        + rng.normal(0, 1.0, size=n)
    )

    cooling_efficiency = (
        1.0
        - 0.03 * (ambient_temp_c - 24)
        - 0.01 * (humidity_pct - 55)
        + 0.1 * (plant == "Plant B").astype(float)
        + rng.normal(0, 0.15, size=n)
    )

    vibration_rms = (
        0.15 * wear_index
        - 0.45 * maintenance_quality
        + 0.08 * (lubrication_interval_days - 14)
        - 0.25 * cooling_efficiency
        + 0.25 * (machine_type == "Conveyor").astype(float)
        + rng.normal(0, 0.6, size=n)
    )

    defect_rate = (
        0.35 * (production_target_pressure)
        - 0.55 * procedure_compliance
        + 0.15 * (machine_type == "Press").astype(float)
        + rng.normal(0, 0.5, size=n)
    )

    failure_risk = sigmoid(
        0.9 * vibration_rms
        + 0.15 * wear_index
        + 0.25 * defect_rate
        + 0.15 * (machine_type == "Press").astype(float)
        + rng.normal(0, 0.6, size=n)
    )

    repair_duration_hours = np.clip(
        6
        + 10 * failure_risk
        - 2.5 * spare_parts_availability
        + 1.5 * (plant == "Plant C").astype(float)
        + rng.normal(0, 2.0, size=n),
        1, 60
    )

    # Outcome: unplanned downtime in next 30 days
    # Has multiple drivers, including multi-hop paths.
    downtime_hours_30d = np.clip(
        2
        + 18 * failure_risk
        + 0.9 * repair_duration_hours
        + 2.5 * defect_rate
        + 0.4 * (machine_type == "Conveyor").astype(float)
        + rng.normal(0, 3.0, size=n),
        0, 200
    )

    df = pd.DataFrame({
        "Plant": plant,
        "Machine Type": machine_type,
        "Shift": shift,
        "Ambient Temperature (C)": ambient_temp_c,
        "Humidity (%)": humidity_pct,
        "Machine Age (Years)": machine_age_years,

        "Maintenance Quality Index": maintenance_quality,
        "Lubrication Interval (Days)": lubrication_interval_days,
        "Operator Training Hours (Monthly)": operator_training_hours,
        "Spare Parts Availability Index": spare_parts_availability,
        "Production Target Pressure Index": production_target_pressure,

        "Fatigue Index": fatigue,
        "Procedure Compliance Index": procedure_compliance,
        "Wear Index": wear_index,
        "Cooling Efficiency Index": cooling_efficiency,
        "Vibration RMS": vibration_rms,
        "Defect Rate Index": defect_rate,
        "Failure Risk (0-1)": failure_risk,
        "Repair Duration (Hours)": repair_duration_hours,

        "Unplanned Downtime Hours (Next 30d)": downtime_hours_30d,
    })

    # Inject some missingness
    cols = df.columns.tolist()
    for c in cols:
        if c == "Unplanned Downtime Hours (Next 30d)":
            continue
        mask = rng.random(n) < missing_rate
        df.loc[mask, c] = np.nan

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}  rows={len(df)} cols={df.shape[1]}")
    print("Outcome column:", "Unplanned Downtime Hours (Next 30d)")


if __name__ == "__main__":
    make_synth_manufacturing()
