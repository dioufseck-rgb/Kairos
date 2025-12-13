import numpy as np
import pandas as pd

def make_synthetic_kairos_dataset(
    n: int = 5000,
    seed: int = 42,
    path: str = "/workspaces/Kairos/synth_kairos.csv"
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # -------------------------
    # Uncontrollable / context
    # -------------------------
    region = rng.choice(["Northeast", "Southeast", "Midwest", "Southwest", "West"], size=n, p=[0.22, 0.18, 0.20, 0.15, 0.25])
    segment = rng.choice(["Leisure", "Business", "Commuter"], size=n, p=[0.45, 0.35, 0.20])
    age_range = rng.choice(["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], size=n, p=[0.10, 0.22, 0.20, 0.18, 0.17, 0.13])
    distance_miles = rng.gamma(shape=3.0, scale=80.0, size=n)  # right-tailed
    season = rng.choice(["Winter", "Spring", "Summer", "Fall"], size=n)

    # Encode some context numerically for the data-generating process (hidden from Kairos as "structure")
    region_risk = pd.Series(region).map({
        "Northeast": 0.20, "Southeast": 0.10, "Midwest": 0.15, "Southwest": 0.12, "West": 0.18
    }).values

    segment_pressure = pd.Series(segment).map({
        "Business": 0.25, "Commuter": 0.15, "Leisure": 0.05
    }).values

    season_weather = pd.Series(season).map({
        "Winter": 0.35, "Spring": 0.15, "Summer": 0.10, "Fall": 0.20
    }).values

    # -------------------------
    # Latent factor (unobserved)
    # -------------------------
    # Think: "infrastructure constraint / systemic disruption"
    latent_disruption = rng.normal(0, 1, size=n)

    # -------------------------
    # Observed proxies for latent factor (helps causal discovery + confounding)
    # -------------------------
    # These correlate with latent_disruption but are observed.
    staffing_level = np.clip(0.6 + 0.2*rng.normal(size=n) - 0.15*latent_disruption + 0.05*segment_pressure, 0.2, 1.0)
    track_congestion = np.clip(0.4 + 0.25*rng.normal(size=n) + 0.20*latent_disruption + 0.10*region_risk, 0.0, 1.0)

    # -------------------------
    # Controllable levers
    # -------------------------
    # "Service investments" that influence outcome directly and via mediators
    comms_quality = np.clip(3.4 + 0.6*rng.normal(size=n) + 0.8*staffing_level - 0.3*latent_disruption, 1.0, 5.0)
    cleanliness = np.clip(3.6 + 0.6*rng.normal(size=n) + 0.5*staffing_level, 1.0, 5.0)
    comfort = np.clip(3.5 + 0.7*rng.normal(size=n) + 0.3*staffing_level, 1.0, 5.0)
    wifi = np.clip(3.1 + 0.8*rng.normal(size=n) + 0.4*staffing_level - 0.1*track_congestion, 1.0, 5.0)
    food = np.clip(3.0 + 0.9*rng.normal(size=n) + 0.2*staffing_level, 1.0, 5.0)

    # A “pricing” lever with confounding: business/region/distance influence fare,
    # and those also influence satisfaction via other channels.
    base_fare = 40 + 0.22*distance_miles + 35*segment_pressure + 12*region_risk
    promo = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    total_fare = np.clip(base_fare * (1 - 0.08*promo) + rng.normal(0, 10, size=n), 10, None)

    # -------------------------
    # Mediators (important for causal structure tests)
    # -------------------------
    # Delays are influenced by latent factors + congestion + weather + distance.
    depart_delay = np.clip(
        5
        + 25*season_weather
        + 30*track_congestion
        + 10*region_risk
        + 0.03*distance_miles
        + 12*latent_disruption
        + rng.normal(0, 8, size=n),
        0, None
    )
    arrival_delay = np.clip(depart_delay + 0.4*rng.normal(0, 10, size=n) + 0.02*distance_miles, 0, None)

    # “On-time” perception is a nonlinear transformation (exec-friendly)
    on_time_perf = np.clip(5.0 - 0.045*arrival_delay + 0.15*rng.normal(size=n), 1.0, 5.0)

    # -------------------------
    # Outcome (explanandum)
    # -------------------------
    # Satisfaction depends on:
    # - on-time + comms + cleanliness + comfort + wifi + food (direct)
    # - fare (negative, with interaction: business less price-sensitive)
    # - some residual effect of context + latent (unobserved confounding)
    business_flag = (segment == "Business").astype(float)

    satisfaction = (
        1.2
        + 0.85*on_time_perf
        + 0.35*comms_quality
        + 0.25*cleanliness
        + 0.28*comfort
        + 0.18*wifi
        + 0.10*food
        - 0.006*total_fare*(1 - 0.6*business_flag)     # business customers less price-sensitive
        - 0.18*season_weather
        - 0.10*region_risk
        - 0.25*np.maximum(latent_disruption, 0)         # disruption hurts satisfaction (unobserved)
        + rng.normal(0, 0.55, size=n)
    )
    overall_satisfaction = np.clip(satisfaction, 1.0, 5.0)

    # -------------------------
    # Assemble dataset
    # -------------------------
    df = pd.DataFrame({
        # Uncontrollable
        "Region": region,
        "Customer Segment": segment,
        "Age Range": age_range,
        "Season": season,
        "Distance (Miles)": distance_miles.round(1),

        # Observed proxies (helpful for discovery / confounding)
        "Staffing Level (Proxy)": staffing_level.round(3),
        "Track Congestion (Proxy)": track_congestion.round(3),

        # Controllable levers
        "Total Fare Amount": total_fare.round(2),
        "Communication About Status": comms_quality.round(2),
        "Cleanliness": cleanliness.round(2),
        "Comfort": comfort.round(2),
        "Wi-Fi": wifi.round(2),
        "Food & Beverage": food.round(2),

        # Mediators / ops outcomes
        "Departure Delay (Minutes)": depart_delay.round(1),
        "Arrival Delay (Minutes)": arrival_delay.round(1),
        "On-time Performance": on_time_perf.round(2),

        # Explanandum
        "Overall Satisfaction": overall_satisfaction.round(2),
    })

    # Add some missingness to test your NaN handling
    for col in ["Food & Beverage", "Wi-Fi", "Communication About Status"]:
        mask = rng.random(n) < 0.08
        df.loc[mask, col] = np.nan

    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = make_synthetic_kairos_dataset()
    print(df.head())
    print(f"\nWrote: /workspaces/Kairos/synth_kairos.csv  (rows={len(df)}, cols={df.shape[1]})")
