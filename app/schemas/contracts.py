"""Versioned evidence contracts, independent of HTTP and model providers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CameraView = Literal['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObjectFact(Contract):
    object_id: str
    category: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    heading_rad: float
    range_xy_m: float
    lidar_points: int | None
    speed_label_mps: tuple[float | None, float | None, float | None]
    geometry_valid: bool


class Predicate(Contract):
    predicate_id: str
    kind: Literal["category", "longitudinal", "lateral", "range", "nearest", "farthest", "longest",
                  "relative_left", "relative_right", "ahead_of", "behind_anchor", "vehicle_heading", "nearest_to_anchor", "vertical_rank", "image_relation"]
    value: str
    phrase: str
    source: str
    required_capability: str = "geometry"
    matches: dict[str, bool | None]


class ContrastPlan(Contract):
    status: Literal["unique_in_gt_candidates", "ambiguous"]
    selected_predicate_ids: list[str]
    remaining_object_ids: list[str]
    removal_tests: dict[str, list[str]]
    same_class_count: int
    hard_negatives: list[dict]
    algorithm: str = "minimum-cardinality-safe-plan-v2"


class ImageEvidence(Contract):
    camera_name: int
    camera_label: str
    projected_box_xyxy: tuple[float, float, float, float]
    crop_xyxy: tuple[int, int, int, int]
    source_jpeg_sha256: str
    width: int
    height: int
    pose_timestamp_s: float | None
    camera_trigger_time_s: float | None
    camera_readout_done_time_s: float | None
    most_visible_camera: bool
    artifacts: dict[str, str]
    appearance_box_xyxy: tuple[int, int, int, int] | None = None
    appearance_camera_object_id: str | None = None
    appearance_identity_source: Literal["official_camera_lidar_association", "official_target_point_projection", "native_box_point_correspondence", "official_lidar_box_projection"] | None = None
    projection_touches_boundary: bool | None = None
    target_front_facing: bool | None = None
    appearance_projected_points: int = Field(default=0, ge=0)


class SceneEvidence(Contract):
    scene_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_points: int = Field(gt=0)
    target_points: int = Field(ge=0)
    source_alignment_verified: bool = False


class EvidencePacket(Contract):
    schema_version: str = "0.2"
    method_version: str = "evidence-contrast-v1"
    packet_id: str
    context: str
    frame_timestamp_micros: int
    target_object_id: str
    task: dict
    policy: dict
    target: ObjectFact
    candidates: list[ObjectFact]
    predicates: list[Predicate]
    contrast: ContrastPlan
    views: list[ImageEvidence]
    track: dict
    provenance: dict
    warnings: list[str] = Field(default_factory=list)
    assets: dict[str, str] = Field(default_factory=dict)
    scene: SceneEvidence | None = None
    ready_for_generation: bool = False
    source_view: CameraView | None = None
    source_camera_object_id: str | None = None


VisualColor = Literal["white", "black", "gray", "silver", "red", "blue", "green", "yellow", "orange", "brown", "beige", "purple", "pink"]
VISUAL_COLORS = ("white", "black", "gray", "silver", "red", "blue", "green", "yellow", "orange", "brown", "beige", "purple", "pink")
VISUAL_ATTRIBUTE_VALUES = {
    "color": VISUAL_COLORS,
    "upper_clothing_color": VISUAL_COLORS,
    "lower_clothing_color": VISUAL_COLORS,
    "helmet_color": VISUAL_COLORS,
    "subtype": ("sedan", "suv", "van", "pickup truck", "truck", "bus", "bus section", "excavator", "excavator boom", "excavator body", "motorcycle", "bicycle", "a-frame sign", "construction barricade"),
    "shape": ("round", "rectangular", "triangular", "diamond shaped", "octagonal"),
    "feature": ("roof rack", "roof cargo box", "trailer", "stripes", "orange and white stripes", "red and white stripes", "black and yellow stripes", "warning light", "illuminated headlights", "illuminated taillights", "backpack", "handheld bag", "umbrella", "high visibility vest"),
}


def visual_attribute_schema(schema):
    """Expose the same kind/value pairs that the runtime validator accepts."""
    properties = schema["properties"]
    kind = "kind" if "kind" in properties else "k"
    value = "value" if "value" in properties else "v"
    schema["oneOf"] = [
        {"properties": {kind: {"const": name}, value: {"enum": list(values)}}}
        for name, values in VISUAL_ATTRIBUTE_VALUES.items()]


class VisualAttribute(Contract):
    model_config = ConfigDict(json_schema_extra=visual_attribute_schema)
    kind: Literal["color", "subtype", "shape", "upper_clothing_color", "lower_clothing_color", "helmet_color", "feature"]
    value: Literal["white", "black", "gray", "silver", "red", "blue", "green", "yellow", "orange", "brown", "beige", "purple", "pink",
                   "sedan", "suv", "van", "pickup truck", "truck", "bus", "bus section", "excavator", "excavator boom", "excavator body", "motorcycle", "bicycle", "a-frame sign", "construction barricade",
                   "round", "rectangular", "triangular", "diamond shaped", "octagonal",
                   "roof rack", "roof cargo box", "trailer", "stripes", "orange and white stripes", "red and white stripes", "black and yellow stripes", "warning light", "illuminated headlights", "illuminated taillights", "backpack", "handheld bag", "umbrella", "high visibility vest"]

    @model_validator(mode="after")
    def supported_value(self):
        allowed = VISUAL_ATTRIBUTE_VALUES[self.kind]
        if self.value not in allowed:
            raise ValueError(f"Attribute kind {self.kind!r} does not allow value {self.value!r}. "
                f"Allowed values for this kind: {', '.join(allowed)}. Omit unsupported or uncertain detail.")
        return self


class VehicleColors(Contract):
    body: VisualColor | None = Field(default=None,
        description="Clearly confirmed paint on visible body surfaces, excluding roof, windows, lights and reflections; never infer hidden surfaces. Null if uncertain or no single dominant visible body color.")
    roof: VisualColor | None = Field(default=None,
        description="Clearly confirmed roof paint only; null when not clearly visible. Never infer it from body paint.")


class VisibleLights(Contract):
    headlights: Literal['present', 'absent', 'uncertain'] = 'uncertain'
    taillights: Literal['present', 'absent', 'uncertain'] = 'uncertain'
    complete_region_inspected: bool = False
    reason: str = Field(default='', max_length=300)


class VisualObservation(Contract):
    target_visible: bool = Field(description="The selected annotation unit itself is separately recognizable from visible RGB surfaces, rather than an occluder or neighboring object. Partial visibility may be true; fully hidden or uncertain identity must be false.")
    visibility: Literal["clear", "partial", "occluded", "uncertain"] = "uncertain"
    visibility_reason: str = Field(default="", max_length=300)
    attributes: list[VisualAttribute] = Field(default_factory=list, max_length=6)
    vehicle_colors: VehicleColors | None = Field(default=None, exclude_if=lambda value: value is None,
        description="Separate body and roof paint for vehicles only; null for other categories. Do not mix roof paint into generic color attributes.")
    visible_lights: VisibleLights | None = Field(default=None, exclude_if=lambda value: value is None,
        description="Camera-visible illuminated lights, independently of unknown body paint. Absent means a complete inspectable original-image region was checked for this visible cue, not that physical lights are off/missing. Use uncertain for glare, noise, hidden/unclear regions or ambiguous light ownership.")


class CandidateVisualObservation(VisualObservation):
    candidate_id: str
    absent_features: list[Literal["roof rack", "roof cargo box", "trailer", "stripes", "orange and white stripes", "red and white stripes", "black and yellow stripes", "warning light", "backpack", "handheld bag", "umbrella", "high visibility vest"]] = Field(default_factory=list, max_length=6)


class SceneVisualObservation(Contract):
    candidates: list[CandidateVisualObservation] = Field(default_factory=list, max_length=32)


class GeneratedExpression(Contract):
    expression: str = Field(min_length=1, max_length=400)
    predicate_ids: list[str] = Field(min_length=1)
    visual_attributes: list[VisualAttribute] = Field(default_factory=list, max_length=6)


class GroundResult(Contract):
    candidate_ids: list[str]
    parsed_predicate_ids: list[str]
    unsupported_claims: list[str] = Field(description="Only unsupported assertions actually made by the expression; [] when none. Do not include parsing explanations, supported camera phrases, or things the sentence does not claim.")
    parsed_visual_attributes: list[VisualAttribute] = Field(default_factory=list, max_length=6)


class AgentAction(Contract):
    action: Literal["expand_relations", "generate", "rewrite", "abstain"]
    reason: str = Field(min_length=1, max_length=300)
