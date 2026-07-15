"""Idempotently scaffold the Leukemia Quantum Pipeline workspace on Windows."""

from pathlib import Path


def create_workspace(root: Path = Path(r"D:\Leukemia_Quantum_Pipeline")) -> None:
    for name in ("data", "code", "logs_and_output", "images"):
        (root / name).mkdir(parents=True, exist_ok=True)
    print(f"Workspace ready: {root}")


if __name__ == "__main__":
    create_workspace()
