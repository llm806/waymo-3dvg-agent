"""Model-independent single-frame inputs and 7-DOF supervision, with provenance."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.contracts import Contract, CameraView

Capability = Literal["geometry", "rgb", "temporal"]


class TaskProfile(Contract):
    task_id: str = "waymo-single-frame-object-v1"
    language: Literal["en"] = "en"
    coordinate_frame: Literal["frame vehicle: x forward, y left, z up"] = "frame vehicle: x forward, y left, z up"
    box_order: tuple[str, ...] = ("cx", "cy", "cz", "length", "width", "height", "heading_rad")
    heading_convention: Literal["counterclockwise about +z; local length axis is +x"] = "counterclockwise about +z; local length axis is +x"
    input_capabilities: tuple[Capability, ...] = ("geometry",)
    prediction: Literal["single native 7-DOF box from complete scene and expression"] = "single native 7-DOF box from complete scene and expression"

    @model_validator(mode="after")
    def check_protocol(self):
        if self.box_order != ("cx", "cy", "cz", "length", "width", "height", "heading_rad"):
            raise ValueError("Native box order is fixed; do not silently reorder dimensions")
        if "geometry" not in self.input_capabilities or "temporal" in self.input_capabilities:
            raise ValueError("This protocol requires geometry and one frame; temporal needs a separate protocol")
        if len(set(self.input_capabilities)) != len(self.input_capabilities):
            raise ValueError("Duplicate input capabilities")
        return self


class NativeBox(Contract):
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    heading_rad: float

    @field_validator("size_m")
    @classmethod
    def positive_size(cls, value):
        if any(x <= 0 for x in value):
            raise ValueError("Box dimensions must be positive")
        return value

    def vector(self) -> list[float]:
        return [*self.center_m, *self.size_m, self.heading_rad]


class AssetRef(Contract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CameraAsset(Contract):
    camera_name: int = Field(ge=1, le=5)
    image: AssetRef
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pose_timestamp_s: float | None
    trigger_time_s: float | None
    readout_done_time_s: float | None


class SceneManifest(Contract):
    schema_version: Literal["1.0"] = "1.0"
    scene_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    context: str
    frame_timestamp_micros: int = Field(gt=0)
    official_split: Literal["training", "validation"]
    coordinate_frame: Literal["frame vehicle: x forward, y left, z up"] = "frame vehicle: x forward, y left, z up"
    points: AssetRef
    point_features: tuple[str, ...] = ("x", "y", "z", "intensity", "elongation", "range")
    point_count: int = Field(gt=0)
    sensor_returns: dict[str, int]
    calibration: AssetRef
    cameras: list[CameraAsset]
    conversion: dict
    source_components: dict
    diagnostics: dict

    @model_validator(mode="after")
    def check_scene(self):
        if self.point_features != ("x", "y", "z", "intensity", "elongation", "range"):
            raise ValueError("Point feature order is fixed")
        if {c.camera_name for c in self.cameras} != {1, 2, 3, 4, 5} or len(self.cameras) != 5:
            raise ValueError("Complete five-camera frame required")
        expected = {f"{laser}/{return_id}" for laser in range(1, 6) for return_id in (1, 2)}
        if set(self.sensor_returns) != expected or any(n < 0 for n in self.sensor_returns.values()):
            raise ValueError("Counts must cover five sensors and both returns, without negative counts")
        if sum(self.sensor_returns.values()) != self.point_count:
            raise ValueError("Sensor/return counts disagree with point_count")
        return self


class QualityCertificate(Contract):
    protocol: Literal["oracle-counterfactual-and-independent-calibrated-v1", "independent-blind-audited-v1"] = "independent-blind-audited-v1"
    oracle_checks_passed: Literal[True]
    counterfactual_checks_passed: bool | None
    source_alignment_verified: Literal[True]
    independent_risk_calibration: Literal[True]
    calibration_report: AssetRef
    validation_report: AssetRef

    @model_validator(mode="after")
    def original_protocol_requires_counterfactual(self):
        if self.protocol == "oracle-counterfactual-and-independent-calibrated-v1" and self.counterfactual_checks_passed is not True:
            raise ValueError("Original release protocol requires passed counterfactual checks")
        return self


class TrainingSample(Contract):
    schema_version: Literal["1.0"] = "1.0"
    sample_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    context: str
    frame_timestamp_micros: int = Field(gt=0)
    track_id: str = Field(min_length=1)
    category: Literal["vehicle", "pedestrian", "sign", "cyclist"]
    official_split: Literal["training", "validation"]
    scene: AssetRef
    expression: str = Field(min_length=1, max_length=400)
    source_view: CameraView | None = None
    source_camera_object_id: str | None = None
    task: TaskProfile
    required_capabilities: tuple[Capability, ...]
    target_box: NativeBox
    annotation_run_id: str
    evidence_packet_id: str
    quality: QualityCertificate

    @model_validator(mode="after")
    def check_capabilities(self):
        if not self.required_capabilities or not set(self.required_capabilities) <= set(self.task.input_capabilities):
            raise ValueError("Expression requires modalities unavailable to the training model")
        if self.track_id in self.expression:
            raise ValueError("Source track ID cannot appear in an expression")
        return self
