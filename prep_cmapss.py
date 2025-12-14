# prepare_cmapss_fd001_multisnapshot.py
# ----------------------------------------
# Build a TABULAR dataset from C-MAPSS FD001
# using MULTIPLE fixed snapshot cycles per engine
# to support causal discovery (PC / FCI).
# ----------------------------------------

from pathlib import Path
import pandas as pd


RAW_PATH = Path("/workspaces/Kairos/train_FD001.txt")
OUT_PATH = Path("/workspaces/Kairos/cmapss_fd001_multisnapshot.csv")

# Choose fixed snapshot cycles (tunable)
SNAPSHOT_CYCLES = [30, 40, 50, 60, 70]

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)


def main():
    df = pd.read_csv(
        RAW_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )

    print(f"Loaded raw data: rows={len(df)} engines={df['engine_id'].nunique()}")

    # Compute max cycle per engine
    max_cycles = (
        df.groupby("engine_id", as_index=False)["cycle"]
        .max()
        .rename(columns={"cycle": "max_cycle"})
    )
    df = df.merge(max_cycles, on="engine_id", how="left")

    # Keep engines that survive to max snapshot
    max_snapshot = max(SNAPSHOT_CYCLES)
    df = df[df["max_cycle"] >= max_snapshot].copy()

    # Keep only desired snapshot cycles
    df = df[df["cycle"].isin(SNAPSHOT_CYCLES)].copy()

    # Compute RUL at each snapshot
    df["RUL"] = df["max_cycle"] - df["cycle"]

    # Final cleanup
    df = df.drop(columns=["max_cycle"])
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    print(f"\nSnapshot cycles used: {SNAPSHOT_CYCLES}")
    print(f"Final dataset shape: {df.shape}")
    print("RUL summary:")
    print(df["RUL"].describe())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()

