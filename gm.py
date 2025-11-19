from __future__ import annotations

from pathlib import Path
import pandas as pd
from patient_dataset import PatientStateDataset


def main() -> None:
    dataset_dir = Path(__file__).parent / "data" / "Datasets"

    dataset = PatientStateDataset(dataset_dir / "glucose_insulin_ICU.csv")
    first_traj = next(iter(dataset))
    print(
        f"Trajectory for subject {first_traj.subject_id} / ICU stay {first_traj.icustay_id} "
        f"with {len(first_traj.states)} states."
    )
    print(*first_traj.states, sep="\n\n")


if __name__ == "__main__":
    main()
