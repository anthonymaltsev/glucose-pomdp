from __future__ import annotations

from pathlib import Path

from patient_dataset import PatientStateDataset


def main() -> None:
    dataset_dir = Path(__file__).parent / "data" / "Datasets"

    dataset = PatientStateDataset(dataset_dir / "glucose_insulin_ICU.csv")
    i = 0
    actions_count = {action.name: 0 for action in dataset.get_action_space()}
    iterator = iter(dataset)
    while i < 300:
        transition = next(iterator)
        actions_count[transition.action.name] += 1
        # print(
        #     "Transition:",
        #     f"{transition.state.timestamp} -> {transition.next_state.timestamp}",
        #     f"action={transition.action.name}",
        #     f"reward={transition.reward}",
        #     )
        print("State feature vector:", transition.state.features)
        i += 1
    print(actions_count)


if __name__ == "__main__":
    main()
