"""Read-only access to already-built evidence; opening the UI never calls a model."""

import json
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse

from app.schemas.contracts import EvidencePacket
from app.datasets.evidence import EVIDENCE_ROOT
from app.datasets.evidence import ROOT
from app.infrastructure.artifacts import checked_asset
from app.schemas.training_contracts import SceneManifest

router = APIRouter(prefix="/api/evidence", tags=["evidence"])
PacketId = Annotated[str, Path(pattern=r"^[a-f0-9]{24}$")]


@router.get("")
def index() -> dict:
    path = EVIDENCE_ROOT / "index.json"
    value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"packets": [], "association_audits": [], "errors": []}
    for item in value["packets"]:
        saved = packet(item["packet_id"])
        if (item["context"], item["timestamp"], item["object_id"]) != (
                saved.context, saved.frame_timestamp_micros, saved.target_object_id):
            raise HTTPException(409, "Evidence index identity differs from saved packet")
        item["source_file"] = saved.provenance.get("components", {}).get("lidar_box", {}).get("file")
    return value


@router.get("/{packet_id}", response_model=EvidencePacket)
def packet(packet_id: PacketId) -> EvidencePacket:
    path = EVIDENCE_ROOT / packet_id / "packet.json"
    if not path.exists():
        raise HTTPException(404, "Evidence packet not found")
    return EvidencePacket.model_validate_json(path.read_text(encoding="utf-8"))


@router.get("/{packet_id}/assets/{name}")
def asset(packet_id: PacketId, name: str) -> FileResponse:
    evidence = packet(packet_id)
    if name not in evidence.provenance["artifact_sha256"]:
        raise HTTPException(404, "Unknown evidence asset")
    try:
        return FileResponse(checked_asset(EVIDENCE_ROOT / packet_id,name,evidence.provenance['artifact_sha256'][name]))
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get("/{packet_id}/inspection")
def inspection(packet_id: PacketId) -> dict:
    evidence = packet(packet_id)
    if not evidence.scene:
        raise HTTPException(409, "Official scene not prepared")
    from app.datasets.inspection import inspect_packet
    try:
        return inspect_packet(packet_id, evidence.scene.manifest_sha256)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{packet_id}/links")
def links(packet_id: PacketId) -> dict:
    evidence = packet(packet_id)
    sources = {}
    if evidence.scene:
        try:
            manifest = checked_asset(ROOT, evidence.scene.manifest_path, evidence.scene.manifest_sha256)
            sources = SceneManifest.model_validate_json(manifest.read_text(encoding='utf-8')).source_components
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    from app.datasets.sample_links import sample_links
    return sample_links(evidence, sources)


@router.get("/{packet_id}/scene-assets/{name}")
def scene_asset(packet_id: PacketId, name: str) -> FileResponse:
    evidence = packet(packet_id)
    if not evidence.scene:
        raise HTTPException(404, "Scene not prepared")
    try:
        manifest = checked_asset(ROOT, evidence.scene.manifest_path, evidence.scene.manifest_sha256)
        scene = SceneManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
        image = next((c.image for c in scene.cameras if c.image.path == name), None)
        if image is None:
            raise HTTPException(404, "Unknown scene image")
        return FileResponse(checked_asset(manifest.parent, image.path, image.sha256))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
