from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.datasets.data import DatasetInventory, inspect_dataset

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
dataset_root = Path(__file__).resolve().parents[2] / "waymo_trainset"


@router.get("/inventory", response_model=DatasetInventory)
def inventory() -> DatasetInventory:
    try:
        return inspect_dataset(dataset_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset directory is missing") from exc
