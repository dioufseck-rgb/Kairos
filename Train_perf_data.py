import numpy as np
import pandas as pd

def generate_corridor_data(
    n_trips: int = 5000,
    n_segments: int = 5,
    n_stations: int = 6,
    seed: int = 42,
):
    """
    Synthetic rail corridor performance data inspired by Amtrak delay taxonomy.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_trips):
        trip = {}

        # --- Context ---
        passengers = rng.integers(80, 500)
        trip["passengers"] = passengers
        trip["equipment_type"] = rng.choice(["single_level", "bi_level"])
        trip["dep_hour"] = rng.integers(5, 22)

        cumulative_delay = 0.0
        total_buffer_absorbed = 0.0

        # --- Segments ---
        for g in range(1, n_segments + 1):
            host_rr = rng.gamma(1.5, 2.0)
            signal = rng.gamma(1.2, 1.5)
            track = rng.gamma(1.3, 1.8)
            weather = rng.exponential(1.0)
            mechanical = rng.gamma(1.1, 1.2)
            incident = rng.binomial(1, 0.02) * rng.uniform(10, 40)

            segment_delay = (
                host_rr + signal + track + weather + mechanical + incident
            )

            trip[f"host_rr_delay_g{g}"] = host_rr
            trip[f"signal_delay_g{g}"] = signal
            trip[f"track_delay_g{g}"] = track
            trip[f"weather_delay_g{g}"] = weather
            trip[f"mechanical_delay_g{g}"] = mechanical
            trip[f"incident_delay_g{g}"] = incident
            trip[f"segment_total_delay_g{g}"] = segment_delay

            cumulative_delay += segment_delay

        # --- Stations ---
        for s in range(1, n_stations + 1):
            dwell_passenger = rng.gamma(1.2, 0.5) * (passengers / 200)
            dwell_crew = rng.gamma(1.1, 0.3)
            dispatch = rng.gamma(1.0, 0.6)
            servicing = rng.gamma(1.0, 0.4)
            connection_hold = rng.binomial(1, 0.15) * rng.uniform(2, 8)

            scheduled_buffer = rng.uniform(2, 6)
            buffer_absorbed = min(cumulative_delay, scheduled_buffer)

            arrival_delay = max(
                cumulative_delay
                + dwell_passenger
                + dwell_crew
                + dispatch
                + servicing
                + connection_hold
                - buffer_absorbed,
                0.0,
            )

            trip[f"dwell_passenger_delay_s{s}"] = dwell_passenger
            trip[f"dwell_crew_delay_s{s}"] = dwell_crew
            trip[f"dispatch_delay_s{s}"] = dispatch
            trip[f"servicing_delay_s{s}"] = servicing
            trip[f"connection_hold_delay_s{s}"] = connection_hold
            trip[f"scheduled_buffer_s{s}"] = scheduled_buffer
            trip[f"buffer_absorbed_s{s}"] = buffer_absorbed
            trip[f"arrival_delay_s{s}"] = arrival_delay

            cumulative_delay = arrival_delay
            total_buffer_absorbed += buffer_absorbed

        # --- Outcomes ---
        trip["final_arrival_delay_min"] = cumulative_delay
        trip["OTP_15"] = int(cumulative_delay <= 15)
        trip["passenger_weighted_OTP_15"] = trip["OTP_15"] * passengers

        rows.append(trip)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_corridor_data()
    df.to_csv("synthetic_corridor_trips.csv", index=False)
    print("Generated synthetic_corridor_trips.csv")
