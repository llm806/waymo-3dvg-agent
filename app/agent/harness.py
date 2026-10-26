"""Shared generation prompts and blind native-geometry verification; never releases labels."""

import base64
import json
import io
import random
import re
from pathlib import Path
from PIL import Image

from app.schemas.contracts import EvidencePacket, GeneratedExpression, GroundResult, VisualAttribute
from app.quality.contrast import remaining
from app.quality.visibility_policy import allows_partial, checks_rgb_visibility, usable_rgb_observation
from app.datasets.evidence import EVIDENCE_ROOT, ROOT
from app.infrastructure.artifacts import checked_asset

RUN_ROOT = ROOT / "runtime" / "annotation"
TASK_GUIDE = """Write one clear English referring expression for a person looking out from the ego vehicle.
Help that person pick out exactly one real object using the point-cloud-derived spatial facts
and the confirmed details in the source photograph. Sound like ordinary spoken English.
Use a short noun phrase or a short sentence. Start with a capital letter; preserve SUV casing.
Aim for 10-25 words when possible, keeping whatever is truly needed to distinguish the object.

The supplied facts come from native 3D boxes, point evidence, calibration and identity-checked RGB
observations. They are evidence, not a script to recite. Prefer one of prechecked_reference_options: each bundle was checked against all possible competitors.
Keep its necessary cues while choosing your own natural wording. A true landmark relation by itself
is not necessarily unique. Use the prechecked_unique_cue_plan or an equally unique combination of available_target_facts and confirmed target attributes.
Choose a recognizable core noun and one or two useful details; add a verified relationship to a
salient landmark if it helps. A red sedan, a person wearing a black helmet, or a distinctive sign
can be useful landmarks ONLY when supplied in available_landmarks with a verified relation.
For a confirmed appearance/road-scene policy, retain required_recognition_attributes even if
an ordinal alone is unique. Include the required supported ego direction as scene context.
An expression such as "the leftmost sign" is incomplete when its color/shape is confirmed.
Combine a recognizable noun, the supplied salient details and a useful spatial clause naturally;
do not pad the sentence, list every attribute or invent missing details to satisfy this preference.
Do not add a landmark for decoration if the target is already clear. Avoid chains of relations.
Prefer confirmed appearance plus an ego-relative location, then a short relation to an identifiable
landmark. Use a horizontal image order only when those scene-based facts cannot distinguish it.
Do not use the source camera's name as filler when the sentence needs no image-frame relation.
An ego-relative direction already states the object's scene position. Do not repeat it as
"visible in the left-side view, to the left of the ego vehicle" when a supplied camera-free
bundle is unique. A native 3D landmark relation also needs no camera name. Only image-frame
relations (such as leftmost in a view or image-left of a landmark), or a visibility qualifier
actually needed for uniqueness, require a camera qualifier. Never remove a necessary cue by
assuming that the source image implicitly narrows the complete candidate inventory.
For example, "The white pickup truck ahead and to the left of the ego vehicle" is ordinary phrasing
when those facts are confirmed. This is a wording example, not permission to invent a color or type.
Landmark words describe that landmark, not the target. visual_attributes lists ONLY target details.
Retain every fact needed to identify a landmark, even when expressing it in ordinary English.
Never invent a building, tree, crossing, lane, traffic light, sign wording or helmet from context.
A background object is usable only when it has a supplied verified landmark identity and relation.

Use verified ego directions such as ahead, behind, to the left or to the right. Native 3D relations
to other objects use the ego axes too: x forward, y left, z up, regardless of the other object's heading.
Image relations instead use the explicitly named camera view. Name that view in the sentence,
using its camera_view_bindings: front view, front-left view, front-right view, left-side view or
right-side view. Camera 5 is right-side view, not front-right view. source_view is not an ego direction.
Keep the word visible when a relation compares visible objects only. Keep every comparison-group
qualifier; do not change an attribute-qualified order into a rank over all objects.
Also keep an ALL-OBJECT rank separate from target-only appearance details: adding rectangular
to "the leftmost sign" changes the group to rectangular signs. If shape is target-only, write
"The leftmost sign in the front view is rectangular, ahead and to the left of the ego vehicle."
Do not add "visible" to an all-object comparison; that would change its scope too.
Follow comparison_scope_constraints: place target-only details in a separate clause after
the complete comparison, rather than inserting them inside its ranked noun phrase.
Do not repeat a camera name or add a redundant ego-location clause just to make a longer sentence.
Do not write internal codes (FRONT_LEFT etc.), coordinates, IDs, box numbers, highlights or measurements.
Never use distances, approximate near/far guesses, nearest/closest/farthest/furthest or distance ranks.

Use everyday words. Do not write outline/silhouette extents, pixel edges, native-center height ranks,
highest/lowest-positioned, image-height comparisons, or phrases such as "whose outline reaches lowest",
"extends lower than the others", "takes up the most vertical space", "appearing as", or "same garment colors".
Never write bottommost, bottom-most, bottom most, topmost, lowermost, uppermost, second from the
bottom/top, or a target at the top/bottom/corner of an image. Those describe framing, not a useful
road-scene location. Do not paraphrase them as a lower/higher car, a closest car or a physical height.
Do not rename those technical comparisons as closest, tallest, at the bottom, or any other claim.
If ordinary supported facts cannot distinguish the target, do not hide the ambiguity with a fluent sentence.

Only claim attributes in the canonical confirmed attributes list, not details from visibility_reason prose.
Use a confirmed specific vehicle type (sedan, SUV, pickup truck, excavator) rather than generic vehicle.
Keep separately boxed bus sections, excavator booms and excavator bodies separate from the whole machine.
A confirmed A-frame sign means a folding sign unit, not an adjacent barrier or a sign on one pole.
When required_core_type is supplied, use that recognizable core type as the noun and list its subtype.
When required_vehicle_body_color is supplied, include that body color in both expression and
visual_attributes; roof paint, glass and lights cannot substitute for body paint.
For people, name confirmed garment regions explicitly: black top, blue bottoms, etc. Do not guess
jacket/trouser/skirt type when only garment color was observed. "In a black top" and "wearing a black top"
are equivalent. Vague "in black" cannot stand for separately claimed upper and lower clothing colors.
Use helmet_color only for an actual confirmed worn helmet: "wearing a black helmet"; not hair, a hood,
skin or shirt color. Holding/carrying a bag and wearing a backpack describe different equipment.
For a cyclist, "on a bicycle" describes equipment; "riding" adds unsupported motion.
Do not guess make/model, identity, parked/stationary status, movement, intent or future actions.
Partial occlusion does not prevent a description when the actual native unit is recognizable.
Describe its visible confirmed features; never borrow the foreground occluder's appearance.

Follow required_predicate_ids. expression_style is a soft writing preference, not a fixed opening.
Vary wording only when it remains natural and preserves the facts. Do not mechanically fill every
attribute/location/relation slot or repeat the subject, clothing, accessories or camera qualifier.
Return JSON with expression, predicate_ids and visual_attributes. Every target attribute actually
spoken must be listed; no unspoken attributes. Predicate IDs are metadata, never sentence content.
On repair, address the feedback using verified evidence and preserve the necessary distinguishing cues."""

SCENE_TASK_GUIDE = """Write one natural English referring expression for a person observing traffic from the ego vehicle.
Identify one real native 3D object using its confirmed identity, appearance and verified scene position.
Start with its most specific confirmed noun (sedan, SUV, van, pickup truck, bus section, sign, person).
Keep required_recognition_attributes: at most three salient confirmed details such as body color,
roof rack, trailer, clothing, helmet or carried equipment. Add the required supported ego direction.
Use an independently recognizable supplied landmark only when needed to distinguish the target.
Prefer landmarks with a confirmed distinctive type, color, shape or equipment visible in the source image.
Use one short verified relation such as ahead of, behind or to the left/right of that landmark;
two relations are allowed only when the complete uniqueness proof actually needs both.
Prefer a prechecked_reference_options bundle; preserve all its distinguishing cues in your own words.
A fluent sentence is not a uniqueness proof. Never invent missing attributes to make it distinctive.

Do not use image order: leftmost, rightmost, first/second from either side, edge/corner positions,
or paraphrases such as the far-left vehicle. Do not replace them with nearest/farthest or pixel/height ranks.
Do not use mere camera membership or "visible in a view" as a referring cue. The source photograph
is evidence, not a location description. Omit camera names for ego-relative/native 3D relations.
An explicitly supplied image-landmark relation still needs its exact camera; image-left/right is not ego-left/right.
Use only supplied verified landmarks and their descriptor/relationship. Do not invent trees, buildings,
road names or other free landmarks from the photo. Do not use an unidentifiable generic neighbor.

Describe only the confirmed native unit: bus sections and excavator booms/bodies are not the whole machine.
Body paint is separate from roof paint, windows and lights. Use required_vehicle_body_color when supplied.
For people, name confirmed garment regions (a black top, blue bottoms) and actual worn helmet colors.
A carried bag and a backpack differ; "on a bicycle" describes equipment, "riding" asserts motion.
Partial occlusion is acceptable when the actual unit is recognizable; never borrow an occluder's appearance.
Use only canonical confirmed attributes, never details guessed from visibility_reason or the context photo.
Do not guess make/model, text on a sign, identity, motion, intent, parking status or future actions.
Never state IDs, coordinates, box numbers, distances, distance ranks, pixel extents or artificial markings.

Follow required_predicate_ids. Use an object-first phrase or short sentence, usually 10-25 words;
retain extra words only when needed for the verified distinction. Avoid duplicate attributes and clauses.
Return JSON with expression, predicate_ids and visual_attributes. List every target attribute actually
spoken and no unspoken target attributes. Landmark attributes belong to the supplied landmark predicate.
On repair use the feedback and confirmed evidence without reviving a forbidden image-order cue."""

MINIMAL_TASK_GUIDE = """Write one natural English sentence or noun phrase that uniquely identifies the native 3D object.
Truthful, supported claims and unique grounding are the acceptance criteria. Use the simplest prechecked
reference option in your own words. Prefer a specific confirmed type and a salient intrinsic feature;
use ego-relative position or a recognizable landmark when helpful. There is no compulsory attribute
count, ego-direction clause, landmark clause, opening template or word count. Do not add unnecessary facts.
When intrinsic appearance is unique in the actual source image and its native 3D identity is verified,
an image_attribute fact is a complete referring cue: retain its exact camera scope and spoken attributes.
For example, a confirmed uniquely identifiable red sedan in the front view needs no ranking or extra landmark.
Prefer such appearance references to leftmost/rightmost. A verified horizontal relation/order is a fallback
when needed for a unique reference, not the default writing template.
An image_boundary fact can describe a recognizable partial object cut off at an original image edge.
Treat image-edge truncation and image ordering as last-resort disambiguators, never as the default description.
First use confirmed salient properties/specific category and useful verified position relative to the ego vehicle,
or a distinctive verified landmark. Prefer a supplied scene-based reference option over a boundary option.
Keep the verified spatial qualifiers inside a landmark description: they may distinguish two similar
landmarks. Do not drop them or transfer an ego-direction clause from the landmark onto the target.
Retain helpful confirmed details such as a roof rack, clothing or vehicle type instead of reducing the subject
to a generic colored object. Do not force any attribute or ego clause when unknown or unnecessary.
Only if the ordinary scene options cannot distinguish it, use the actual supplied edge and exact camera.
Partial visibility alone is never a rejection, and camera-edge contact does not prove closeness to the ego vehicle.
Never invent an attribute, landmark,
motion or intent; never substitute camera-left/right for ego-left/right. Distances and IDs are not cues.
Describe the confirmed native unit, including separately boxed vehicle parts, rather than an adjoining unit.
Return JSON with expression, predicate_ids and visual_attributes. Include only claims actually spoken,
and all claims needed by the chosen uniqueness proof. Landmark attributes are part of that landmark predicate.
On repair correct the reported truth, parsing or uniqueness failure using the available evidence."""

TRAFFIC_TASK_GUIDE = """Write one natural English referring expression as a person describing city traffic.
Identify the confirmed native object with its salient intrinsic features, a useful verified position
relative to the ego vehicle, or a verified relation to an independently identifiable landmark.
Choose the simplest prechecked option that uniquely identifies it; no fixed attribute count, color,
subtype, ego clause, landmark or sentence template is compulsory. Keep useful confirmed details.
Describe objects and their real scene relationships, not the framing of a photograph. Never describe
an object as partly cut off, cropped, clipped, truncated, outside the frame, or touching an image edge.
Image-edge contact and partial visibility are evidence metadata, not referring cues in the sentence.
Do not relabel truncation as closeness to the ego vehicle, a lane position, or a road-edge location.
Partial visibility does not disqualify an otherwise recognizable object. If ordinary verified cues
cannot uniquely identify it, request further evidence or another view/frame; never invent a substitute.
For an intrinsic appearance fact scoped to its source camera, retain the exact camera qualifier and
all distinguishing attributes. A verified horizontal image relation/order is only a necessary fallback;
image-left/right is not ego-left/right. Never add unverified attributes, motion, intent, parking status,
distances, IDs, or arbitrary buildings/trees. Describe the confirmed native unit, not an adjoining unit.
Keep the exact verified descriptor and spatial qualifiers of each supplied landmark; do not transfer
the landmark's ego direction onto the target. Preserve every claim necessary for the uniqueness proof.
Return JSON with expression, predicate_ids and visual_attributes. List all and only the claims actually
spoken. On repair replace the unsupported or unnatural cue with another fully verified unique option;
never just delete a necessary distinguishing clause from the previous sentence."""

GROUND_GUIDE = """You are an independent blind reader of an outdoor 3D referring expression.
No candidate is specially designated. These are oracle native-box candidates, not a real point-cloud test.
Infer only claims actually expressed in the sentence; map them to the symmetric predicate vocabulary.
Use the ego vehicle coordinate frame: x forward, y left, z up. Select ALL candidates possibly matching;
return [] if none match. Do not force a unique answer or infer missing qualifiers. Similar/tied candidates
must all be returned. Report numeric distances, unsupported appearance, motion, subtype, identity or exact-coordinate claims.
Never add attributes merely because the selected candidate has them. For example, "the nearest vehicle"
states category and relative distance rank; it does not state absolute near range or left/right position.
For lowest/highest-positioned within an explicitly named region, parse both the compound vertical_rank
predicate and each expressed region qualifier. Rank box center.z within that region, not the whole scene;
unknown region membership remains possible. Height position is not object height/size.
Return JSON with candidate_ids, parsed_predicate_ids, unsupported_claims. Never use source IDs."""

CLAIM_PARSE_GUIDE = """Parse the literal claims of one outdoor 3D referring expression.
You receive only the sentence, coordinate convention and scene-wide symmetric predicate vocabulary.
No target, generation declarations, object coordinates or expected answer are provided.
Map ALL actually expressed claims to predicate IDs; do not infer missing attributes.
Equivalent ego phrases map to the same literal predicate: ahead of/in front of ->
longitudinal:front; behind/to the rear of -> longitudinal:behind; to the left/right of
or on the left/right side of the ego vehicle -> lateral:left/right.
"Ahead and to the left/right of the ego vehicle" explicitly expresses BOTH front and lateral
predicates; "behind and to the left/right" likewise expresses BOTH behind and lateral.
For example, nearest does not imply left/right or an absolute distance bin.
vehicle_heading/image_vehicle_heading predicates mean directly in front of or directly behind
the LANDMARK vehicle along its own heading, not ego-forward or image-left/right. Preserve the
complete landmark descriptor and any named camera. Do not replace these with ego directions.
Lowest/highest-positioned means center.z order in the explicitly named region, not object height.
Parse the compound vertical_rank and each explicitly expressed region qualifier.
Return literal color, subtype, shape, clothing and feature claims separately in parsed_visual_attributes
using its schema. These are independently checked against associated images; never infer them.
Garment colors use upper_clothing_color or lower_clothing_color, never generic color.
A jacket/coat/shirt describes upper clothing; trousers/pants/skirt describe lower clothing.
An explicitly colored helmet maps to helmet_color, never color or upper_clothing_color.
For example "the person wearing a black helmet" states category:pedestrian and helmet_color:black.
The same helmet phrase on a cyclist states helmet_color without changing category:cyclist.
"In a black top" and "wearing a black top" both state upper_clothing_color:black.
"A black top" states upper_clothing_color:black; "black bottoms" states lower_clothing_color:black.
The generic garment words top/bottoms do not assert a particular jacket/trouser/skirt type.
Person and pedestrian both imply category:pedestrian. Holding/carrying a bag maps to
feature:handheld bag when explicitly held/carried at the hand/side; a worn backpack maps to
feature:backpack. These are visible equipment claims, not walking or crossing intentions.
Vague "black clothing" or "dressed in black" does not explicitly name both garment regions;
report ambiguous clothing scope in unsupported_claims instead of inventing two region claims.
For registered image_relation predicates, require the explicitly named camera in the text.
Attribute-qualified image orders (e.g. leftmost black vehicle) require the entire
descriptor, not just leftmost. Also parse those target attributes separately in
parsed_visual_attributes; they are target claims, not landmark attributes.
Attribute-qualified image orders can also rank image extents/heights within a clothing group.
The same rule applies to any explicitly described group, such as round signs, vehicles
with roof racks, people holding bags or cyclists on bicycles. Preserve each named qualifier
in the parsed scope and target attributes; missing equipment never proves equipment absence.
"Among visible pedestrians wearing a black top in the front-left view,
the one that extends lower than the others" states the CLOTHING-QUALIFIED bottom_edge_visible rule,
category:pedestrian and upper_clothing_color:black. It does not rank every pedestrian in that image.
Attributes explicitly defining this comparison group also apply to the referred member.
When the referred pedestrian explicitly wears a black top and black bottoms, a later comparison
to visible pedestrians "with the same garment colors" ranks that same upper/lower color group.
Record BOTH target garment attributes separately, as well as the clothing-qualified order.
Image orders ending in _visible rank only visibly recognizable objects in the named camera.
Parse these ONLY when the sentence explicitly says visible; an unqualified leftmost/rightmost
claim cannot exclude hidden objects and must not be mapped to a _visible rule.
"The visible sign at the bottom/top of the left-side view" maps to the named camera's
bottommost_visible/topmost_visible rule, not to ego height or distance. Explicit visible scope is required.
For bottom_edge_visible/top_edge_visible, "extends lower/higher than the others" and
"extending lowest/highest" in an explicitly named
camera means the visible image extent, not center.z, height, distance or an ego-frame rank.
Equivalent relative clauses such as "the visible vehicle whose outline reaches lowest/highest
in the front view" express the same image-extent rule. They do not express object height or distance.
The height_visible rule's "tallest visible outline" in a named camera means RGB image height;
it does not imply physical object height, ego distance, or a native 3D height rank.
"Whose visible outline is tallest" and "takes up the most vertical space" in the named view
express this same height_visible rule. In "Among visible [group] in [view], the [member] extends
lower/higher than the others", all and only the attributes stated in [group] scope the edge comparison.
Attributes stated only on [member] are additional target claims and do not narrow that group.
For example, "Among visible vehicles in the left-side view, the blue sedan extends lower than
the others" maps to image_order:4:vehicle:bottom_edge_visible plus target color:blue and
subtype:sedan. It must NOT map to vehicle~color=blue~subtype=sedan:bottom_edge_visible:
that narrower group was never stated. In "Among visible sedans in the left-side view, the blue
one extends lower than the others", the comparison scope is subtype:sedan ONLY, not color:blue.
In "Among visible blue sedans in the left-side view, the one that extends lower than the others",
the scope includes both color:blue and subtype:sedan. Preserve this distinction for all
categories, attributes, image heights and cameras; target details after the group never
silently restrict the comparison group.
An explicit camera qualifier at the start of the sentence scopes its image-order clause just
as a qualifier after that clause does. Parse equivalent grammatical word orders identically;
ego directions remain ego-relative even within a camera-qualified sentence.
Natural view names (front view, front-left view, front-right view, left-side view,
right-side view) denote their corresponding cameras; they are not ego directions.
Unqualified left/right stays in the ego frame. Never infer a camera from metadata or vocabulary.
Appearance words inside a registered landmark description belong to that predicate, not the target:
parsed_visual_attributes must contain ONLY the referred target's appearance. Report any unsupported
change to the landmark descriptor as unsupported_claims.
Sedan/SUV/van/pickup truck/truck/bus/excavator imply category:vehicle; subtype is also a visual attribute.
Excavator is a supported construction vehicle subtype, not an unsupported category or motion claim.
An A-frame sign implies category:sign and subtype:a-frame sign. It describes the folding sign unit,
not sign wording or an adjacent plastic construction barrier.
A construction barricade implies category:sign and subtype:construction barricade, referring to the
selected native traffic barricade unit. Orange-and-white striped (or orange and white stripes)
means feature:orange and white stripes; likewise red and white stripes or black and yellow stripes.
These phrases do not claim a single dominant color. A warning light means feature:warning light;
it does not establish flashing or any other motion. Parse these supported features literally.
Illuminated headlights (headlights on) means feature:illuminated headlights; illuminated taillights
(lit red rear lights) means feature:illuminated taillights. These do not claim body paint or motion.
Preserve an explicitly named camera for camera-visible light descriptions. Never turn image-right
in the front-left camera into the ego vehicle's right, or replace front lamps with rear lamps.
Bus section, excavator boom and excavator body refer to separately boxed native units; each also
implies category:vehicle. Parse the exact spoken subtype, without expanding it to the whole machine.
"On a bicycle" and "with a bicycle" explicitly express subtype:bicycle; do not mark that supported
schema attribute as unsupported. "Cyclist" alone states category:cyclist without entailing a visual
subtype or motion. Verbs such as riding/moving, and speed claims, remain unsupported motion claims.
Report numeric distances (including approximate or spelled-out measurements), behavior, motion, identity, exact coordinates and other unsupported claims in unsupported_claims.
unsupported_claims contains ONLY unsupported assertions actually made by the sentence; return [] when none exist.
Do not put parsing explanations, cautions, statements about what you did NOT infer, or supported camera phrases in this list.
A literal claim that maps to a supplied predicate or the visual-attribute schema is supported for parsing;
its actual truth and uniqueness are checked by the downstream geometry and image tools, not by this reader.
For example, "visible in the front view" maps to the registered front-camera visibility predicate;
this camera interpretation is not an unsupported claim or an ego-direction inference.
Do not calculate spatial relations or choose an object; a deterministic geometry tool does that.
Return JSON with candidate_ids: [], parsed_predicate_ids, parsed_visual_attributes and unsupported_claims.
Always include parsed_visual_attributes. List every target attribute literally expressed in the subject
or comparison group; return [] only when the text states no target appearance attributes."""


# This local guard is independent of the model's claim parser and declarations.
MINIMAL_CLAIM_PARSE_GUIDE = """Independently parse the literal claims of one 3D referring expression.
Use only the supplied scene-wide predicate vocabulary and attribute schema. No target is designated.
Map all and only spoken facts. Do not infer an ego direction, subtype, color or landmark from context.
Ahead/in front of the ego vehicle means longitudinal:front; behind means longitudinal:behind;
left/right of the ego vehicle means lateral:left/right. An explicitly coordinated direction states both.
Sedan/SUV/van/truck/bus imply category:vehicle AND a separate subtype visual attribute.
Never invent predicate IDs for visual attributes; use only IDs in the supplied vocabulary.
When a chosen image_order/image_attribute ID explicitly includes color/subtype/clothing qualifiers,
those qualifiers apply to the referred member too: repeat them in parsed_visual_attributes.
For example, leftmost visible beige sedan states BOTH color:beige and subtype:sedan; do not return [].
vehicle_heading/image_vehicle_heading means directly in front of or directly behind the landmark
vehicle along that vehicle's own heading. This is distinct from ego-forward and image-left/right.
image_joint_heading has the same geometry with an indefinite landmark descriptor: a matching
landmark need not be unique alone. Parse the complete spoken descriptor and camera; its color
and subtype belong to the landmark, not the target. Do not infer a relation absent from the text.
image_joint_spatial describes a target relative to a matching indefinite landmark of ANY supported category.
native_left/right/ahead/behind follow the EGO vehicle's orientation and use native 3D box extents,
not camera coordinates or the landmark heading. Their ego orientation must be explicit in the sentence.
native_above/below require 3D vertical separation and horizontal footprint alignment, not image height.
Its reference_frame is mandatory: image_left/right/above/below are SOURCE IMAGE axes; higher_level/lower_level
are disjoint native vertical levels. Do not replace one with another or with ego directions. Landmark
appearance fields describe the landmark, not the target. Parse every clause in a conjunction separately.
An image_joint_heading ahead/rearward predicate says only in front of/behind along the landmark
heading, without direct alignment. front/rear requires directly/immediately; never add that meaning.
Preserve its complete landmark descriptor, direct alignment and any named camera; never infer them.
An image_attribute phrase identifies an appearance in its explicitly named camera. Match the entire
spoken descriptor and camera; also report its category and target attributes separately. A view name
alone does not assert image_attribute, image order or visible. Parse image_visibility only when visible
is actually stated. Camera-left/right is different from ego-left/right.
An image_boundary predicate requires the stated cut-off edge and named camera; do not infer it from
the word foreground, apparent size, an ordinary derived crop, or partial visibility without a named edge.
Color/subtype/shape/equipment clauses describing the subject are target visual attributes. Attributes
inside a landmark noun phrase belong to the landmark predicate, not to the subject. Clothing colors
use upper_clothing_color/lower_clothing_color; helmet colors use helmet_color. Person means pedestrian.
Use only facts actually named; report an unsupported or ambiguous claim instead of inventing a mapping.
Return JSON with candidate_ids: [], parsed_predicate_ids, parsed_visual_attributes, unsupported_claims.
Do not choose an object or calculate uniqueness; the deterministic tools do that after parsing."""

_DISTANCE_MEASUREMENT = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|half|quarter|a|an)"
    r"[\s-]*(?:m|km|cm|mm|met(?:er|re)s?|kilomet(?:er|re)s?|centimet(?:er|re)s?|"
    r"millimet(?:er|re)s?|feet|foot|ft|yards?|yd|inches?|car[ -]lengths?|vehicle[ -]lengths?)\b"
    r"|\d+(?:\.\d+)?\s*(?:公里|千米|厘米|毫米|米|英尺|码)", re.IGNORECASE)


_DISTANCE_RANK = re.compile(r"\b(?:nearest|closest|farthest|furthest)\b|最近的|最远的", re.IGNORECASE)
_FIRST_LETTER = re.compile(r'''^[\s"'“‘]*([A-Za-z])''')


def capitalize_expression(expression: str) -> str:
    text = expression.strip()
    first = _FIRST_LETTER.match(text)
    return text[:first.start(1)] + first[1].upper() + text[first.end(1):] if first else text


def normalize_expression(expression: str) -> str:
    """Render camera enum references as speech, without changing the viewpoint."""
    from app.datasets.waymo import CAMERAS
    from app.quality.description_search import CAMERA_VIEWS
    for camera, code in sorted(CAMERAS.items(), key=lambda item: -len(item[1])):
        name = re.escape(code).replace('_', '[_ -]')
        expression = re.sub(r'\b' + name + r'\s+camera\s+(?:image|view)\b',
            CAMERA_VIEWS[camera], expression, flags=re.IGNORECASE)
    return capitalize_expression(expression)


def repair_camera_qualifier(packet: EvidencePacket, generated: GeneratedExpression):
    """Propose one explicit local wording repair; acceptance still needs two blind checks."""
    from app.quality.description_search import CAMERA_VIEWS, confirmed_image_camera
    selected = [p for p in packet.predicates if p.predicate_id in generated.predicate_ids and p.kind == 'image_relation']
    if not selected or any(p.matches.get(packet.target_object_id) is not True for p in selected):
        return generated, None
    cameras = {int(p.predicate_id.split(':')[1]) for p in selected}
    if len(cameras) != 1 or not packet.provenance.get('admission_plan', {}).get('passed'):
        return generated, None
    camera = cameras.pop()
    if camera not in {v.camera_name for v in packet.views} or not confirmed_image_camera(packet, camera):
        return generated, None
    mentioned = [(number, match) for number, name in CAMERA_VIEWS.items()
        for match in re.finditer(r'\b' + re.escape(name).replace(r'\-', '[- ]') + r'\b', generated.expression, re.IGNORECASE)]
    if not mentioned and not re.search(r'\b(?:view|camera|image|frame|photo(?:graph)?)\b',generated.expression,re.IGNORECASE):
        # Complete a missing declared, verified scope in a NEW draft only. Do
        # not reinterpret ambiguous existing image wording or a second camera.
        needs_visible=any(p.predicate_id.startswith('image_visibility:') for p in selected)
        text=generated.expression.rstrip()
        if text.endswith(('!','?')):
            return generated,None
        ending='.' if text.endswith('.') else ''
        body=text[:-1] if ending else text
        add_visible=needs_visible and not re.search(r'\bvisible\b',text,re.IGNORECASE)
        expression=body+(', visible in the ' if add_visible else ' in the ')+CAMERA_VIEWS[camera]+ending
        if len(expression)>400:
            return generated,None
        repair={'kind':'missing_verified_camera_qualifier','before':generated.expression,'after':expression,
            'predicate_ids':[p.predicate_id for p in selected],'requires_two_blind_checks':True}
        return generated.model_copy(update={'expression':expression}),repair
    if len(mentioned) != 1 or mentioned[0][0] == camera:
        return generated, None
    _, match = mentioned[0]
    expression = generated.expression[:match.start()] + CAMERA_VIEWS[camera] + generated.expression[match.end():]
    repair = {'kind':'registered_camera_qualifier', 'before':generated.expression, 'after':expression,
        'predicate_ids':[p.predicate_id for p in selected], 'requires_two_blind_checks':True}
    return generated.model_copy(update={'expression':expression}), repair


def repair_comparison_scope(packet: EvidencePacket, generated: GeneratedExpression):
    """Move target-only adjectives out of one explicit extreme comparison.

    This proposes a new draft; it is never a verifier shortcut. Restrict the
    rewrite to an exact, known noun phrase and a proven registered order. All
    facts/attributes stay declared, the original output remains in its trace,
    and both blind readers must validate the resulting sentence.
    """
    if not packet.task.get('description_detail_policy') or packet.target.category not in {'vehicle','sign'}:
        return generated, None
    from app.quality.description_search import CAMERA_VIEWS, appearance_phrase, image_order_attributes
    orders = [p for p in packet.predicates if p.predicate_id in generated.predicate_ids
        and p.predicate_id.startswith('image_order:')]
    if len(orders)!=1 or orders[0].matches.get(packet.target_object_id) is not True:
        return generated, None
    predicate=orders[0]
    _,camera,_,order=predicate.predicate_id.split(':')
    if camera not in {str(c) for c in CAMERA_VIEWS} or order not in {'leftmost','rightmost','leftmost_visible','rightmost_visible'}:
        return generated, None
    group=image_order_attributes(predicate)
    extras=[a for a in generated.visual_attributes if a not in group]
    if (not extras or any(a.kind not in {'color','shape'} for a in extras)
            or any(a not in generated.visual_attributes for a in group)):
        return generated, None
    noun=appearance_phrase(packet.target.category,generated.visual_attributes)[4:]
    # Match only an object-first phrase with this exact named camera; no fuzzy
    # edits, partial attribute guesses, other relations or finished records.
    pattern=r'^The '+order.removesuffix('_visible')+r' (?:visible )?'+re.escape(noun)+r'(?: visible)? in the '+re.escape(CAMERA_VIEWS[int(camera)])
    match=re.match(pattern,generated.expression,re.IGNORECASE)
    if match is None:
        return generated, None
    subject='The '+order.removesuffix('_visible')+(' visible ' if order.endswith('_visible') else ' ')
    subject+=appearance_phrase(packet.target.category,group)[4:]+' in the '+CAMERA_VIEWS[int(camera)]
    adjectives=' and '.join(a.value for a in sorted(extras,key=lambda a:a.kind!='color'))
    tail=generated.expression[match.end():]
    if tail.startswith(' is '):
        expression=subject+' is '+adjectives+' and '+tail[4:]
    elif tail=='.' or tail.startswith(', '):
        expression=subject+' is '+adjectives+tail
    else:
        return generated, None
    if len(expression)>400:
        return generated, None
    repair={'kind':'separate_target_adjectives_from_comparison_group','before':generated.expression,'after':expression,
        'predicate_ids':[predicate.predicate_id],'requires_two_blind_checks':True}
    return generated.model_copy(update={'expression':expression}),repair


def expression_policy_errors(expression: str) -> list[str]:
    errors = []
    if re.search(r'\b(?:FRONT|FRONT_LEFT|FRONT_RIGHT|SIDE_LEFT|SIDE_RIGHT)\b', expression):
        errors.append('internal_camera_code_in_text')
    if re.search(r'[.!?]\s+\S', expression):
        errors.append('multiple_sentences_in_expression')
    if _DISTANCE_MEASUREMENT.search(expression):
        errors.append("numeric_distance_in_text")
    if _DISTANCE_RANK.search(expression):
        errors.append("distance_rank_in_text")
    first = _FIRST_LETTER.match(expression)
    if first and first[1].islower():
        errors.append("sentence_initial_not_capitalized")
    return errors

def expression_language_errors(expression: str) -> list[str]:
    """Repair new drafts only; never retroactively change saved export eligibility."""
    errors = []
    if re.search(r'\b(?:outline|silhouette|pixels?|bounding box)\b|\b(?:highest|lowest)[ -]positioned\b|\bvertical space\b|\b(?:extends?|extending|reaches) (?:lower|higher|lowest|highest)\b', expression, re.IGNORECASE):
        errors.append('abstract_spatial_wording')
    if re.search(r'\bwhose (?:visible )?outline (?:reaches|is)\b|\bsame garment colors\b', expression, re.IGNORECASE):
        errors.append('mechanical_referring_expression')
    if re.search(
            r'\b(?:bottom|top|lower|upper)[\s-]*most\b'
            r'|\b(?:highest|lowest)\b'
            r'|\b(?:first|second|third|fourth|fifth)\s+from\s+(?:the\s+)?(?:top|bottom)\b'
            r'|\b(?:highest|lowest|higher|lower|upper)\s+(?:visible\s+)?(?:vehicle|car|sedan|suv|van|truck|bus|sign|pedestrian|cyclist)\b'
            r'|\b(?:higher|lower)\s+than\b'
            r'|\b(?:at|near|towards?|in)\s+(?:the\s+)?(?:very\s+)?(?:top|bottom|upper|lower)'
            r'(?:[ -]+(?:left|right))?(?:\s+(?:edge|corner|part|portion|half|side))?\s+(?:of|in)\s+'
            r'(?:the\s+)?(?:[\w-]+\s+){0,4}(?:view|image|frame|photo(?:graph)?|picture|screen)\b'
            r'|\b(?:upper|lower)[ -]+(?:left|right)[ -]+corner\b',expression,re.IGNORECASE):
        errors.append('image_vertical_or_corner_wording')
    # Catch copied appearance clauses, not a word-count limit that could erase
    # necessary evidence. Shared short direction words are not repetition.
    words = re.findall(r"[a-z]+", expression.casefold())
    seen = set()
    for index in range(len(words) - 5):
        phrase = tuple(words[index:index + 6])
        if phrase in seen and set(phrase) & {'wearing', 'holding', 'carrying', 'top', 'bottoms', 'rack', 'cargo'}:
            errors.append('repeated_appearance_description')
            break
        seen.add(phrase)
    return errors


def repair_redundant_camera_qualifier(packet, generated, observations):
    """Simplify a new candidate, never a saved annotation or its acceptance.

    Only recognized camera-visibility clauses are edited. The shorter candidate
    must already have a complete uniqueness proof and still undergo both blind
    checks. Other wording is left for the model rather than guessed at by regex.
    """
    if not packet.task.get('natural_description'):
        return generated, None
    from app.quality.description_search import CAMERA_VIEWS, without_redundant_camera_visibility, expression_candidates
    catalog = {p.predicate_id:p for p in packet.predicates}
    if any(id_ not in catalog for id_ in generated.predicate_ids):
        return generated, None
    chosen = [catalog[id_] for id_ in generated.predicate_ids]
    reduced = without_redundant_camera_visibility(packet,chosen,generated.visual_attributes,observations or {})
    camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view),None)
    view = CAMERA_VIEWS.get(camera)
    if not view:
        return generated, None
    view_pattern = re.escape(view).replace(r'\-',r'[- ]')
    qualifier = r'\b(?:in|from)\s+the\s+' + view_pattern + r'\b'
    if len(re.findall(qualifier,generated.expression,re.IGNORECASE)) != 1:
        return generated, None
    if len(reduced) == len(chosen):
        if any(p.kind == 'image_relation' for p in chosen):
            return generated, None
        proof = expression_candidates(packet,chosen,generated.visual_attributes,observations or {})
        if proof['native'] != [packet.target_object_id] or proof['image_only']:
            return generated, None
    text = re.sub(r'\b(?:(?:that|which)\s+is\s+)?visible\s+(?=' + qualifier + ')',
        '',generated.expression,flags=re.IGNORECASE)
    text = re.sub(qualifier + r',?\s*','',text,flags=re.IGNORECASE)
    text = re.sub(r'^\s*(the)\s+visible\s+',r'\1 ',text,flags=re.IGNORECASE)
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'\s+,',',',text)
    text = re.sub(r',\s*,',',',text)
    text = re.sub(r',\s*([.!?])',r'\1',text)
    text = normalize_expression(text.strip())
    if (not text or re.search(r'\b(?:view|camera|image|visible)\b',text,re.IGNORECASE)
            or re.search(r'\b(?:is|are|was|be|clearly|only)\s*[.!?]?$',text,re.IGNORECASE)):
        return generated, None
    ids = [p.predicate_id for p in reduced]
    return generated.model_copy(update={'expression':text,'predicate_ids':ids}), {
        'before':generated.expression,'after':text,'original_predicate_ids':generated.predicate_ids,
        'removed_predicate_ids':[p.predicate_id for p in chosen if p not in reduced],
        'reason':'camera-free expression uniquely identifies the target in the complete native/image inventory',
        'requires_two_blind_checks':True}


def normalize_visual_predicate_aliases(packet, generated):
    """Remove only redundant appearance aliases already declared as attributes.

    The sentence and its claims are unchanged. Unknown spatial IDs and aliases
    without the identical visual attribute still fail; both blind rounds remain
    mandatory. Preserve the raw declarations in the repair receipt.
    """
    catalog = {p.predicate_id for p in packet.predicates}
    aliases = {f'{a.kind}:{a.value}' for a in generated.visual_attributes}
    removed = [id_ for id_ in generated.predicate_ids if id_ not in catalog and id_ in aliases]
    if not removed:
        return generated, None
    ids = list(dict.fromkeys(id_ for id_ in generated.predicate_ids if id_ not in removed))
    return generated.model_copy(update={'predicate_ids':ids}), {
        'original_predicate_ids':generated.predicate_ids,'removed_duplicate_visual_ids':removed,
        'reason':'identical claims remain in visual_attributes; no text or evidence changed',
        'requires_two_blind_checks':True}


def normalize_parsed_scope_attributes(predicates, parsed):
    """Complete redundant reader fields from that reader's own compound claims.

    No target, writer declarations or observations enter this function. An
    independently parsed 'leftmost beige sedan' entails beige and sedan for its
    referred member. Only absent kinds are filled; explicit contradictory values
    survive and must fail ordinary agreement/appearance/uniqueness checks.
    """
    from app.quality.description_search import image_order_attributes
    selected = set(parsed.parsed_predicate_ids)
    required = {}
    for predicate in predicates:
        if predicate.kind == 'image_relation' and predicate.predicate_id in selected:
            for attribute in image_order_attributes(predicate):
                required.setdefault(attribute.kind, {}).setdefault(attribute.value, []).append(predicate.predicate_id)
    known = {a.kind for a in parsed.parsed_visual_attributes}
    attributes = list(parsed.parsed_visual_attributes)
    additions = []
    for kind, values in sorted(required.items()):
        if kind in known or kind != 'feature' and len(values) != 1:
            continue
        for value, sources in sorted(values.items()):
            attribute = VisualAttribute(kind=kind,value=value)
            attributes.append(attribute)
            additions.append({'attribute':attribute.model_dump(),'parsed_predicate_ids':sources})
    if not additions:
        return parsed, None
    return parsed.model_copy(update={'parsed_visual_attributes':attributes}), {
        'policy':'compound-claim-entailment-v1','additions':additions,
        'scope':'semantic entailment of independently parsed compound predicates; no target-derived facts'}


def image_scope_errors(packet, expression, predicates, *, check_order=False):
    """The same explicit image-scope checks before and after blind parsing."""
    from app.datasets.waymo import CAMERAS
    from app.quality.description_search import CAMERA_VIEWS
    text=' '.join(expression.lower().replace('_',' ').replace('-',' ').split())
    errors=[]
    for predicate in predicates:
        if predicate.kind!='image_relation':
            continue
        camera=int(predicate.predicate_id.split(':')[1])
        if predicate.predicate_id.startswith('image_visibility:') and not re.search(r'\bvisible\b',text):
            errors.append('image_visibility_not_explicit_in_text')
        if check_order and predicate.predicate_id.startswith('image_joint_spatial:'):
            from app.quality.spatial_relations import BY_KEY
            relation=BY_KEY.get(json.loads(predicate.value).get('relation'))
            # An early omission guard only; the fresh blind parses still check
            # attribute binding, scope and every actual claim in the sentence.
            if relation is None or not re.search(r'\b'+re.escape(relation.words)+r'\s+(?:a|an)\b',text):
                errors.append('declared_joint_relation_not_in_text')
            if relation is not None and relation.frame=='ego_native' and not re.search(
                    r"\bego vehicle(?:'s)? (?:orientation|axes|coordinate frame)\b",text):
                errors.append('native_relation_frame_not_explicit')
        if check_order and predicate.predicate_id.startswith('image_joint_heading:'):
            direction = json.loads(predicate.value)['direction']
            relation = (r'\b(?:directly|immediately)\s+(?:in\s+front\s+of|ahead\s+of)\s+(?:a|an)\b'
                if direction == 'front' else r'\b(?:directly|immediately)\s+behind\s+(?:a|an)\b')
            if not re.search(relation,text):
                if direction not in {'ahead','rearward'}:
                    errors.append('declared_joint_relation_not_in_text')
            if direction in {'ahead','rearward'}:
                ordinary = r'\b(?:in\s+front\s+of|ahead\s+of)\s+(?:a|an)\b' if direction=='ahead' else r'\bbehind\s+(?:a|an)\b'
                stronger = r'\b(?:directly|immediately)\s+(?:in\s+front\s+of|ahead\s+of|behind)\s+(?:a|an)\b'
                if not re.search(ordinary,text) or re.search(stronger,text):
                    errors.append('declared_joint_relation_not_in_text')
        if check_order and predicate.predicate_id.startswith('image_order:'):
            order = predicate.predicate_id.rsplit(':',1)[-1]
            if order.endswith('_visible') and not re.search(r'\bvisible\b',text):
                errors.append('image_visibility_scope_not_explicit')
            # Reject an obviously stale rank declaration before another paid
            # blind parse; this is rejection only, never automatic acceptance.
            direction = next((side for side in ('left','right') if order.startswith(
                (side+'most','second_from_'+side))), None)
            if direction:
                rank = (r'\bsecond\s+from\s+(?:the\s+)?'+direction+r'\b' if order.startswith('second') else
                    r'\b(?:'+direction+r'\s*most|(?:far|extreme)\s+'+direction+r')\b')
                if not re.search(rank,text):
                    errors.append('declared_image_order_not_in_text')
        name=CAMERAS[camera].lower().replace('_',' ')
        natural=CAMERA_VIEWS[camera].replace('-',' ')
        if not (re.search(r'\b'+re.escape(name)+r'\s+(?:camera|image)\b',text)
                or re.search(r'\b'+re.escape(natural)+r'\b',text)):
            errors.append('image_camera_not_explicit_in_text')
        if 'rgb' not in packet.task.get('training_input_capabilities',[]):
            errors.append('image_relation_requires_rgb')
    return errors


def draft_reference_errors(packet, generated, observations=None):
    """New drafts cannot smuggle a retired cue through a fluent paraphrase."""
    if not packet.task.get('natural_description'):
        return []
    from app.quality.description_search import draft_reference_predicate, scene_reference_policy, minimal_reference_policy, without_redundant_camera_visibility
    chosen = set(generated.predicate_ids)
    predicates = [p for p in packet.predicates if p.predicate_id in chosen]
    errors = (['non_scene_reference_cue'] if any(not draft_reference_predicate(packet,p) for p in predicates) else [])
    if not chosen or chosen - {p.predicate_id for p in packet.predicates}:
        errors.append('invalid_predicate_reference')
    from app.quality.description_search import traffic_scene_policy
    if traffic_scene_policy(packet) and image_framing_description(generated.expression):
        errors.append('image_framing_not_traffic_description')
    if packet.task.get('annotation_search') and not errors:
        errors.extend(image_scope_errors(packet,generated.expression,predicates,check_order=True))
    if scene_reference_policy(packet) and not minimal_reference_policy(packet) and re.search(
            r'\b(?:left[- ]?most|right[- ]?most|(?:first|second|third|last)\s+from\s+(?:the\s+)?(?:left|right)|'
            r'(?:far|extreme)[- ](?:left|right)|(?:left|right)[- ]hand[- ](?:edge|end))\b',
            generated.expression, re.IGNORECASE):
        errors.append('image_order_not_scene_reference')
    if scene_reference_policy(packet) and not minimal_reference_policy(packet) and re.search(r'\bvisible\s+in\b',generated.expression,re.IGNORECASE):
        errors.append('camera_visibility_not_scene_reference')
    if not any(p.kind == 'image_relation' for p in predicates) and re.search(
            r'\b(?:front(?:[- ](?:left|right))?|(?:left|right)[- ]side)\s+(?:view|camera|image)\b',
            generated.expression,re.IGNORECASE):
        errors.append('camera_qualifier_without_image_relation')
    if len(without_redundant_camera_visibility(packet,predicates,generated.visual_attributes,observations or {})) < len(predicates):
        errors.append('redundant_camera_qualifier')
    return errors


def image_framing_description(expression):
    """Catch camera-framing prose even when it omits its boundary predicate.

    Road edges, curbs and physical occlusion remain legitimate scene facts;
    partial visibility stays in evidence and does not reject the object.
    """
    text = expression.casefold().replace('-', ' ')
    return bool(re.search(
        r'\b(?:cut\s*off|cropp?ed|clipped|truncated|out\s+of\s+(?:the\s+)?(?:frame|view|image))\b|'
        r'\b(?:partly|partially)\s+outside\s+(?:the\s+)?(?:\w+\s+){0,3}(?:view|image|frame)\b|'
        r'\b(?:edge|border|boundary)\s+of\s+(?:the\s+)?(?:\w+\s+){0,4}(?:view|image|frame|camera|picture)\b|'
        r'\b(?:view|image|frame|camera|picture)\s+(?:(?:bottom|top|left|right)\s+)?(?:edge|border|boundary)\b', text))


def claim_payload(packet: EvidencePacket, expression: str, seed: int) -> tuple[dict, dict[str, str]]:
    payload, mapping = blind_payload(packet, expression, seed)
    del payload["candidates"]
    random.Random(seed).shuffle(payload["predicate_vocabulary"])
    payload["grounding_mode"] = "blind_claim_parse_then_native_geometry"
    from app.quality.description_search import minimal_reference_policy
    if minimal_reference_policy(packet):
        from app.quality.description_search import draft_reference_predicate
        allowed = {p.predicate_id for p in packet.predicates if draft_reference_predicate(packet,p)}
        payload['predicate_vocabulary'] = [p for p in payload['predicate_vocabulary'] if p['predicate_id'] in allowed]
        payload['appearance_reference_scope'] = ('An image_attribute predicate names an intrinsic appearance '
            'reference in its explicit camera. Parse it when that complete appearance and camera are spoken; '
            'also report its literal target attributes and category. Never add an unspoken appearance, '
            'ego direction or landmark. Camera scope alone does not imply an image_attribute predicate.')
    return payload, mapping


def verify_tool_grounding(packet: EvidencePacket, generated: GeneratedExpression, parsed: GroundResult,
                          mapping: dict[str, str], appearance_observations: dict | None = None,
                          target_appearance_verified: bool = False) -> dict:
    """Language claims are blind; numeric candidate filtering belongs to tools.

    Keep any supplied model selection separately. It is never relabelled as a
    successful VLM grounding, and old strict baseline verification is unchanged.
    """
    raw_parsed = parsed
    scope_entailment = None
    if packet.task.get('claim_normalization_policy') == 'compound-claim-entailment-v1':
        parsed, scope_entailment = normalize_parsed_scope_attributes(packet.predicates, parsed)
    def closure(ids):
        by_id = {p.predicate_id: p for p in packet.predicates}
        expanded = set(ids)
        # The compound rule's named region and category are semantic entailments,
        # not facts inferred from the expected target or generator declarations.
        for id_ in ids:
            predicate = by_id.get(id_)
            if predicate is not None and predicate.kind in {'nearest', 'farthest', 'longest', 'nearest_to_anchor'}:
                # Ranking over a named class necessarily entails that class.
                # This reads the public predicate definition, never the target.
                category = predicate.value.split(':')[0]
                implied_category = f'category:{category}'
                if implied_category in by_id:
                    expanded.add(implied_category)
            if (predicate is not None and predicate.kind == 'image_relation'
                    and id_.startswith(('image_order:', 'image_visibility:', 'image_attribute:'))):
                # "The rightmost vehicle in the right-side view" already names
                # the class. Read it from the public order predicate, never the
                # target; image landmark relations do not entail target class.
                parts = id_.split(':')
                if len(parts) == (4 if parts[0] == 'image_order' else 3):
                    implied_category = f'category:{parts[2].split("~")[0]}'
                    if implied_category in by_id:
                        expanded.add(implied_category)
                    if parts[0] == 'image_order' and parts[3].endswith('_visible'):
                        visible = f'image_visibility:{parts[1]}:{parts[2].split("~")[0]}'
                        if visible in by_id:
                            expanded.add(visible)
            if predicate is not None and predicate.kind == "vertical_rank":
                parts = id_.split(":")
                if len(parts) == 6:
                    _, _, category, longitudinal, lateral, distance = parts
                    implied = [f"category:{category}", f"longitudinal:{longitudinal}", f"lateral:{lateral}", f"range:{distance}"]
                    expanded.update(p for p in implied if p in by_id)
        return sorted(expanded)
    normalized_generated = generated.model_copy(update={"predicate_ids": closure(generated.predicate_ids)})
    normalized_parsed = parsed.model_copy(update={"parsed_predicate_ids": closure(parsed.parsed_predicate_ids)})
    feedback = verify(packet, normalized_generated, normalized_parsed, mapping)
    errors = [e for e in feedback["errors"] if e != "blind_reground_not_unique_target"]
    from app.quality.description_search import traffic_scene_policy
    if traffic_scene_policy(packet) and image_framing_description(generated.expression):
        errors.append('image_framing_not_traffic_description')
    from app.quality.description_search import confirmed_image_camera
    expressed = set(generated.predicate_ids) | set(parsed.parsed_predicate_ids)
    from app.quality.description_search import scene_reference_policy, draft_reference_predicate
    if scene_reference_policy(packet) and any(p.predicate_id in expressed
            and not draft_reference_predicate(packet,p) for p in packet.predicates):
        errors.append('non_scene_reference_cue')
    if any(p.kind == 'image_relation' and p.predicate_id in expressed
            and not confirmed_image_camera(packet, int(p.predicate_id.split(':')[1])) for p in packet.predicates):
        errors.append('unconfirmed_image_reference_view')
    if checks_rgb_visibility(packet):
        selected_image = [p for p in packet.predicates if p.kind == 'image_relation' and p.predicate_id in expressed]
        if selected_image:
            # A saved predicate is not a visibility certificate. Rebuild its
            # proof using current camera-bound observations before accepting it.
            from app.quality.description_search import image_relation_catalog
            current_image, _ = image_relation_catalog(packet, appearance_observations or {})
            current_by_id = {p.predicate_id: p for p in current_image}
            if any(p.predicate_id not in current_by_id
                    or p.matches != current_by_id[p.predicate_id].matches for p in selected_image):
                errors.append('unconfirmed_image_reference_evidence')
    native_heading = [p for p in packet.predicates if p.kind == 'vehicle_heading' and p.predicate_id in expressed]
    if native_heading:
        from app.quality.relations import relation_catalog
        rebuilt, _ = relation_catalog(packet, appearance_observations or {})
        current_by_id = {p.predicate_id:p for p in rebuilt}
        if any(p.predicate_id not in current_by_id or p.matches != current_by_id[p.predicate_id].matches
                for p in native_heading):
            errors.append('unconfirmed_vehicle_heading_relation')
    declared_visual = {(a.kind, a.value) for a in generated.visual_attributes}
    parsed_visual = {(a.kind, a.value) for a in parsed.parsed_visual_attributes}
    from app.quality.description_search import image_order_attributes
    for predicate in packet.predicates:
        if predicate.predicate_id in expressed:
            scoped = {(a.kind, a.value) for a in image_order_attributes(predicate)}
            if not scoped <= declared_visual & parsed_visual:
                errors.append('image_order_scope_not_explicit')
            if (predicate.predicate_id.startswith('image_order:')
                    and predicate.predicate_id.endswith('_visible')
                    and not re.search(r'\bvisible\b', generated.expression, re.IGNORECASE)):
                errors.append('image_visibility_scope_not_explicit')
    if declared_visual != parsed_visual or len(declared_visual) != len(generated.visual_attributes):
        errors.append("visual_claims_disagree_with_text")
    if packet.task.get('admission_policy') or packet.provenance.get('admission_plan'):
        if not declared_visual and not packet.task.get('annotation_search'):
            errors.append('prechecked_visual_cue_missing')
        from app.agent.appearance import filter_visual_candidates
        native_ids = {o.object_id for o in packet.candidates}
        if not packet.provenance.get('scene_image_candidates'):
            errors.append('scene_image_inventory_missing')
        image_ids = [c['object_id'] for c in packet.provenance.get('scene_image_candidates', [])
            if c['category'] in {packet.target.category, 'unknown'} and c['object_id'] not in native_ids]
        image_remaining, _ = filter_visual_candidates(image_ids, parsed.parsed_visual_attributes if target_appearance_verified else [],
            appearance_observations or {})
        if image_remaining:
            errors.append('image_expression_not_unique')
    if declared_visual and ("rgb" not in packet.task.get("training_input_capabilities", [])
                            or not appearance_views(packet)):
        errors.append("visual_evidence_unavailable")
    joint_candidates = feedback['symbolic_candidates']
    visual_exclusions = {}
    if (packet.task.get('contrastive_appearance') and appearance_observations is not None
            and target_appearance_verified and declared_visual == parsed_visual and declared_visual):
        from app.agent.appearance import filter_visual_candidates
        joint_candidates, visual_exclusions = filter_visual_candidates(
            joint_candidates, parsed.parsed_visual_attributes, appearance_observations)
        if joint_candidates == [packet.target_object_id]:
            errors = [e for e in errors if e != 'parsed_expression_not_symbolically_unique']
        elif 'parsed_expression_not_symbolically_unique' not in errors:
            errors.append('parsed_expression_not_symbolically_unique')
    search_proof = None
    if packet.task.get('annotation_search') and target_appearance_verified and declared_visual == parsed_visual:
        from app.quality.description_search import expression_candidates
        catalog = {p.predicate_id: p for p in packet.predicates}
        selected = [catalog[id_] for id_ in normalized_parsed.parsed_predicate_ids if id_ in catalog]
        if re.search(r'\b(?:leftmost|rightmost|topmost|bottommost)\b', generated.expression, re.IGNORECASE) and not any(
                p.kind == 'image_relation' and p.predicate_id.startswith('image_order:') for p in selected):
            errors.append('unverified_image_order')
        errors.extend(image_scope_errors(packet,generated.expression,selected))
        search_proof = expression_candidates(packet, selected, parsed.parsed_visual_attributes, appearance_observations or {})
        joint_candidates = search_proof['native']
        errors = [e for e in errors if e not in {'parsed_expression_not_symbolically_unique', 'image_expression_not_unique'}]
        if joint_candidates != [packet.target_object_id]:
            errors.append('parsed_expression_not_symbolically_unique')
        if search_proof['image_only']:
            errors.append('image_expression_not_unique')
        used_anchors = {catalog[id_].value for id_ in normalized_parsed.parsed_predicate_ids
            if id_ in catalog and catalog[id_].kind in {'relative_left', 'relative_right', 'ahead_of', 'behind_anchor', 'vehicle_heading'}}
        for anchor in packet.provenance.get('relation_anchors', []):
            if anchor['anchor_id'] in used_anchors and not (appearance_observations or {}).get(anchor['object_id'], {}).get('target_visible'):
                errors.append('landmark_not_visually_confirmed')
    visual_feedback = ({'declared_visual_attributes': [a.model_dump() for a in generated.visual_attributes],
        'parsed_visual_attributes': [a.model_dump() for a in parsed.parsed_visual_attributes],
        'repair_instruction': 'Make sentence and visual_attributes agree exactly. Name each confirmed garment explicitly (e.g. black jacket and black trousers); do not repeat vague clothing phrasing. Omit unnecessary appearance claims when verified geometry/image cues already identify the target. Never add an unobserved attribute.'}
        if 'visual_claims_disagree_with_text' in errors else {})
    return {**feedback, **visual_feedback, "errors": errors, "diagnostic_pass": not errors,
        **({'raw_parsed_visual_attributes':[a.model_dump() for a in raw_parsed.parsed_visual_attributes],
            'parsed_visual_attributes':[a.model_dump() for a in parsed.parsed_visual_attributes],
            'scope_attribute_entailment':scope_entailment} if scope_entailment else {}),
        "tool_grounded_candidates": joint_candidates,
        "visual_exclusions": visual_exclusions,
        "appearance_observed_candidate_count": len(appearance_observations or {}),
        **({'multimodal_candidate_proof': search_proof} if search_proof is not None else {}),
        "raw_declared_predicate_ids": generated.predicate_ids, "raw_parsed_predicate_ids": parsed.parsed_predicate_ids,
        "claim_normalization": "named-class rank, camera order and vertical-region entailment closure; no target-derived facts",
        "raw_vlm_grounding": {"provided": bool(parsed.candidate_ids),
            "correct_target": feedback["blind_candidates"] == [packet.target_object_id] if parsed.candidate_ids else None,
            "candidate_ids": parsed.candidate_ids, "role": "diagnostic only; not authoritative grounding"},
        "grounding_source": ("native geometry plus explicit-camera 2D relations and verified 2D/3D identity"
            if search_proof and search_proof['camera_identity_proof'] else
            "native geometry plus separately observed contrastive appearance" if visual_exclusions else
            "deterministic native geometry from blindly parsed claims"),
        "protocol": "blind-claim-parse-tool-ground-v1; oracle native boxes; independent quality not certified"}


def declared_cue_failure(packet, generated, observations):
    """Reject ambiguous declared semantics before paying to parse the sentence.

    A successful final parse must agree with these declarations. Even perfect
    agreement cannot rescue an ambiguous combination. This necessary-condition
    check can only reject; it never substitutes for either blind language check.
    """
    if not packet.task.get('annotation_search'):
        return None
    ideal_parse = GroundResult(candidate_ids=[], parsed_predicate_ids=generated.predicate_ids,
        parsed_visual_attributes=generated.visual_attributes, unsupported_claims=[])
    feedback = verify_tool_grounding(packet, generated, ideal_parse, {}, observations, True)
    ambiguity = {'parsed_expression_not_symbolically_unique', 'image_expression_not_unique'}
    if not ambiguity.intersection(feedback['errors']):
        return None
    return {'diagnostic_pass': False, 'errors': ['declared_expression_not_unique'],
        'tool_grounded_candidates': feedback['tool_grounded_candidates'],
        'multimodal_candidate_proof': feedback.get('multimodal_candidate_proof'),
        'protocol': 'local declared-cue necessary-condition check; no language parse performed',
        'repair_instruction': 'The declared cues still match multiple or unknown competitors. '
            'Rephrasing the same cues cannot resolve this. Use a different confirmed distinguishing '
            'attribute or verified landmark/camera relation, preferably the prechecked unique cue plan. '
            'Never invent attributes or delete unknown competitors.'}


def blind_payload(packet: EvidencePacket, expression: str, seed: int = 17) -> tuple[dict, dict[str, str]]:
    objects = list(packet.candidates)
    random.Random(seed).shuffle(objects)
    mapping = {f"c{i:03d}": o.object_id for i, o in enumerate(objects)}
    candidates = [{"candidate_id": label, "category": o.category, "center_m": o.center_m,
                   "size_m": o.size_m, "heading_rad": o.heading_rad, "range_xy_m": o.range_xy_m}
                  for label, o in zip(mapping, objects)]
    # Catalog covers the whole scene, not just target-true predicates.
    vocabulary = [{"predicate_id": p.predicate_id, "phrase": p.phrase} for p in packet.predicates]
    return {"expression": expression, "coordinate_frame": packet.task["coordinate_frame"],
            "candidates": candidates, "predicate_vocabulary": vocabulary,
            "uncertainty_policy": packet.policy}, mapping


def verify(packet: EvidencePacket, generated: GeneratedExpression, grounded: GroundResult,
           mapping: dict[str, str]) -> dict:
    errors = expression_policy_errors(generated.expression)
    by_id = {p.predicate_id: p for p in packet.predicates}
    declared = set(generated.predicate_ids)
    parsed = set(grounded.parsed_predicate_ids)
    from app.quality.contrast import unsupported_ego_directions
    if unsupported_ego_directions(packet.target, declared | parsed):
        errors.append('ego_direction_not_spatially_supported')
    if any(by_id[p].kind in {"range", "nearest", "farthest", "nearest_to_anchor"} for p in declared | parsed if p in by_id):
        errors.append("prohibited_distance_predicate")
    if not declared or not declared <= by_id.keys() or not parsed <= by_id.keys():
        errors.append("invalid_predicate_reference")
    if declared != parsed:
        errors.append("text_claims_disagree_with_declared_predicates")
    if not set(description_predicates(packet)) <= declared:
        errors.append("planned_distinguishing_cue_missing")
    if any(by_id[p].matches[packet.target_object_id] is not True for p in declared if p in by_id):
        errors.append("declared_fact_false_or_unknown")
    if grounded.unsupported_claims:
        errors.append("unsupported_text_claim")
    symbolic = remaining(packet.candidates, [by_id[p] for p in parsed if p in by_id])
    if symbolic != [packet.target_object_id]:
        errors.append("parsed_expression_not_symbolically_unique")
    mapped = [mapping.get(label) for label in grounded.candidate_ids]
    if mapped != [packet.target_object_id]:
        errors.append("blind_reground_not_unique_target")
    if any(o.object_id in generated.expression for o in packet.candidates):
        errors.append("source_object_id_in_text")
    return {"diagnostic_pass": not errors, "errors": errors,
            "declared_predicate_ids": sorted(declared), "parsed_predicate_ids": sorted(parsed),
            "symbolic_candidates": symbolic, "blind_candidates": mapped,
            "unsupported_claims": grounded.unsupported_claims,
            "protocol": "oracle-candidate; common-model bias and projection quality not independently audited"}


def description_predicates(packet: EvidencePacket) -> list[str]:
    from app.quality.description_search import minimal_reference_policy
    if minimal_reference_policy(packet):
        return [f'category:{packet.target.category}']
    selected = ([f'category:{packet.target.category}'] if packet.task.get('natural_description')
                else list(packet.contrast.selected_predicate_ids))
    if packet.task.get("execution_policy") == "evidence-fast-v1" or packet.task.get('natural_description'):
        plan = packet.provenance.get('admission_plan', {})
        concise_plan = (packet.task.get('annotation_search') and packet.task.get('natural_description')
            and plan.get('passed') and plan.get('unique_plan_found')
            and not packet.task.get('description_detail_policy'))
        selected.extend(p.predicate_id for p in packet.predicates
            if p.kind in {"longitudinal", "lateral"} and p.matches.get(packet.target_object_id) is True
            and (not concise_plan or p.predicate_id in plan['predicate_ids']))
    return list(dict.fromkeys(selected))


def appearance_views(packet: EvidencePacket):
    return [v for v in packet.views[:2] if "appearance" in v.artifacts and v.appearance_box_xyxy
        and (not packet.task.get('admission_policy') or
             v.camera_label == packet.source_view and v.appearance_camera_object_id == packet.source_camera_object_id)
        and ((v.appearance_identity_source == 'official_lidar_box_projection')
             or (v.appearance_identity_source in {"official_camera_lidar_association", "native_box_point_correspondence"} and v.appearance_camera_object_id)
             or (v.appearance_identity_source == "official_target_point_projection" and v.appearance_projected_points >= 20))]


def observation_image_input(image: bytes) -> tuple[bytes, str, str]:
    """Meet input size limits without substituting or enlarging RGB evidence."""
    with Image.open(io.BytesIO(image)) as decoded:
        width, height = decoded.size
        padded_width = max(width, 10, (height + 199) // 200)
        padded_height = max(height, 10, (width + 199) // 200)
        if (padded_width, padded_height) == (width, height):
            return image, 'image/jpeg', ''
        left, top = (padded_width - width) // 2, (padded_height - height) // 2
        canvas = Image.new('RGB', (padded_width, padded_height), (128, 128, 128))
        canvas.paste(decoded.convert('RGB'), (left, top))
        buffer = io.BytesIO()
        # Lossless encoding preserves every decoded source pixel; source files
        # and hashes stay unchanged. Padding supplies no new object detail.
        canvas.save(buffer, format='PNG')
    note = (f' Input-size padding only: original_rgb_region={[left, top, left + width, top + height]} '
        f'in this {padded_width}x{padded_height} image (right/bottom exclusive). '
        'Gray padding is artificial and not evidence of the object or scene. '
        'Original RGB pixels were not enlarged. Observe only the indicated region; '
        'if those original pixels are unclear, return visibility=uncertain, not clear.')
    return buffer.getvalue(), 'image/png', note


def appearance_messages(packet: EvidencePacket) -> list[dict]:
    """Independent observation: no generated text, claimed attributes or coordinates."""
    from app.schemas.contracts import VisualObservation
    from app.agent.observation_wire import RGBVisualObservation, VehicleVisualObservation
    from app.agent.appearance import OBSERVATION_SCOPE_GUIDE
    schema = VehicleVisualObservation if packet.target.category == 'vehicle' else RGBVisualObservation
    content = [{"type": "text", "text": json.dumps({"output_schema": schema.model_json_schema(),
        'expected_category': packet.target.category,
        'identity_instruction': 'Observe the target of this category inside the final tight crop. '
            'The full scene and wider crop are context only. If the tight crop shows an occluder or a different '
            'category, report target_visible=false; never describe that other object as the target. '
            'For a padded image, only its indicated original RGB region belongs to the evidence.'})}]
    for view in appearance_views(packet):
        strict = checks_rgb_visibility(packet)
        kinds = ['scene', 'appearance_context', 'appearance'] if strict else ['appearance']
        for kind in kinds:
            if kind not in view.artifacts:
                raise ValueError('Visibility evidence lacks raw scene/context; prepare the target again')
            name = view.artifacts[kind]
            image = checked_asset(EVIDENCE_ROOT / packet.packet_id, name, packet.provenance['artifact_sha256'][name]).read_bytes()
            image, mime, note = observation_image_input(image)
            content.extend([{'type': 'text', 'text': f'Target {kind}, camera {view.camera_label}; identity source: {view.appearance_identity_source}. '
                f'Original camera image: {view.width}x{view.height}; '
                f'selected unit projection touches original camera boundary: {view.projection_touches_boundary}. '
                'The red rectangle in the context image marks the selected unit; the tight crop shows its detail. '
                'An official 3D projection is a region, not an instance mask: do not describe an occluder or neighboring object. '
                'Neither a rectangle nor projected LiDAR returns prove the object is visible. '
                'If the target itself cannot be recognized separately, target_visible=false.' + note},
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(image).decode()}}])
    from app.agent.appearance import visibility_instruction
    return [{"role": "system", "content": OBSERVATION_SCOPE_GUIDE + visibility_instruction(packet) + "Observe the object in the supplied identity-checked target crops. "
             "Independently report ONLY clearly visible attributes using the schema. No guessing from surroundings, "
             "shadow, highlights or occluders. If multiple objects make identity unclear, target_visible=false. "
             "For a vehicle, report vehicle_colors with separate dominant body paint and roof paint; use null for an uncertain part. "
             "When body paint is clear, also include that SAME body color in attributes as kind=color; "
             "both fields are required to confirm it. Never put only the roof color in attributes. "
             "Generic color attributes describe body paint only, never roof paint, windows, lights or reflections. "
             "If roof paint differs from body paint, keep the roof value only under vehicle_colors.roof. "
             "Do not force a dominant body color when none is clear. Clothing colors are not vehicle paint. Omit ambiguous colors/subtypes "
             "and attributes that disagree across views. For visibility, clear means the actual object is recognizable, unoccluded and not cut off; partial means partly hidden or cut off, occluded means fully hidden, uncertain means unclear. Give a short visibility_reason from original RGB surfaces and occluders. Never infer a visible surface from a rectangle or LiDAR dot pattern. Return JSON; do not reason about position or motion."},
            {"role": "user", "content": content}]


def generation_messages(packet: EvidencePacket, previous: dict | None, observations: dict | None = None,
                        *, failed_expressions=()) -> list[dict]:
    from app.agent.appearance import required_body_color
    from app.agent.expression_rendering import generation_style
    selected = set(description_predicates(packet))
    from app.quality.description_search import draft_reference_predicate, scene_reference_policy, minimal_reference_policy, scene_first_policy
    from app.quality.admission import _cue_readability, reference_options, reference_fallback_tier
    options = reference_options(packet, observations or {}, selected) if packet.task.get('annotation_search') else []
    catalog = {p.predicate_id:p for p in packet.predicates}
    scene_options = [option for option in options if not any(
        reference_fallback_tier(catalog[id_]) for id_ in option['predicate_ids'])]
    prefer_scene = bool(scene_first_policy(packet) and scene_options)
    if prefer_scene:
        options = scene_options
    camera_free = [option for option in options if not any(
        catalog[id_].kind == 'image_relation' for id_ in option['predicate_ids'])]
    prefer_camera_free = bool(packet.task.get('natural_description') and camera_free)
    if prefer_camera_free:
        options = camera_free
    option_ids = {id_ for option in options for id_ in option['predicate_ids']}
    plan_ids = set(packet.provenance.get('admission_plan', {}).get('predicate_ids', []))
    plan = packet.provenance.get('admission_plan')
    if prefer_camera_free or prefer_scene:
        # This is a separately proven writer brief, not an edit to frozen evidence.
        plan = {**(plan or {}), **options[0], 'passed':True, 'unique_plan_found':True}
        plan_ids = set(plan['predicate_ids'])
    if packet.task.get('natural_description') and plan and any(
            p.predicate_id in plan_ids and not draft_reference_predicate(packet,p) for p in packet.predicates):
        # Do not leak a historical technical plan back into a new writer's
        # options after filtering its vocabulary. Stored evidence stays intact.
        plan = None
        plan_ids = set()
    facts = [p for p in packet.predicates if p.matches.get(packet.target_object_id) is True
        and p.kind not in {'range', 'nearest', 'farthest', 'nearest_to_anchor'}
        and (not prefer_camera_free or not p.predicate_id.startswith('image_visibility:'))
        and (not prefer_scene or not reference_fallback_tier(p))
        and (not packet.task.get('natural_description') or draft_reference_predicate(packet,p))]
    facts.sort(key=lambda p: (p.predicate_id not in selected | plan_ids | option_ids, _cue_readability(p)))
    facts = [p for index, p in enumerate(facts) if index < 32 or p.predicate_id in selected | plan_ids | option_ids]
    anchor_ids = {p.value for p in facts if p.kind in {'relative_left', 'relative_right', 'ahead_of', 'behind_anchor', 'vehicle_heading'}
        or p.predicate_id.startswith(('image_anchor:', 'image_vehicle_heading:'))}
    landmarks = [a for a in packet.provenance.get('relation_anchors', [])
        if a['anchor_id'] in anchor_ids and a['object_id'] != packet.target_object_id]
    payload = {"task": packet.task, "target": packet.target.model_dump(),
               "source_view": packet.source_view,
               "expression_style": generation_style(packet),
               'camera_view_bindings': [{'predicate_id': p.predicate_id, 'phrase': p.phrase}
                   for p in facts if p.kind == 'image_relation'],
               "source_camera_object_id": packet.source_camera_object_id,
               "required_predicate_ids": sorted(selected),
               "contrast_plan": [{"predicate_id": p.predicate_id, "phrase": p.phrase}
                                 for p in facts if p.predicate_id in selected | plan_ids | option_ids],
               "available_target_facts": [{"predicate_id": p.predicate_id, "phrase": p.phrase}
                    for p in facts],
               "observed_target_appearance": (observations or {}).get(packet.target_object_id),
               "required_vehicle_body_color": required_body_color(packet, observations),
               "available_landmarks": landmarks,
               "prechecked_unique_cue_plan": plan,
               "prechecked_reference_options": options,
               "observed_competitor_appearance": [{'category': obj.category, 'observation': observations[obj.object_id]}
                    for obj in packet.candidates if obj.object_id != packet.target_object_id
                    and obj.category == packet.target.category and observations and obj.object_id in observations],
               "hard_negatives": packet.contrast.hard_negatives,
               "previous_attempt_and_feedback": previous,
               "failed_expressions_do_not_repeat":list(failed_expressions),
               "identity_checked_appearance_inputs": [{"camera": v.camera_label, "source": v.appearance_identity_source}
                   for v in appearance_views(packet)],
               "output_schema": GeneratedExpression.model_json_schema()}
    payload['declaration_contract'] = {
        'allowed_predicate_ids':[p.predicate_id for p in facts],
        'rule':'Select only IDs from available_target_facts. Color/subtype/clothing/equipment go in '
            'visual_attributes, never invented predicate IDs such as subtype:sedan or relation:near_red_car. '
            'After rewriting, rebuild declarations from the NEW sentence; remove every condition no longer spoken. '
            'Do not keep a leftmost/visible rule when replacing its clause with a landmark relation. '
            'Never repeat a failed expression with unchanged declarations. If only declaration metadata was wrong, '
            'correct it from the sentence; otherwise repair the failed wording using a prechecked unique option; '
            'prefer those options over a previously failed ambiguous cue subset.',
        'vehicle_reference_frame':'vehicle_heading and image_vehicle_heading use the LANDMARK vehicle front-face '
            'heading, not ego forward or image left/right. Directly in front/behind requires supplied alignment evidence. '
            'A camera-qualified landmark still requires that exact camera and the complete landmark descriptor.'}
    subtypes = {a['value'] for a in (observations or {}).get(packet.target_object_id, {}).get('attributes', []) if a['kind']=='subtype'}
    payload['required_core_type'] = next(iter(subtypes)) if len(subtypes)==1 and packet.target.category in {'vehicle','sign'} else None
    from app.quality.description_search import descriptive_attributes
    payload['required_recognition_attributes'] = [a.model_dump() for a in descriptive_attributes(packet, observations)]
    if scene_first_policy(packet):
        payload['preferred_recognition_attributes'] = payload['required_recognition_attributes']
        payload['required_recognition_attributes'] = []
    from app.quality.description_search import image_order_attributes
    payload['comparison_scope_constraints'] = [
        {'predicate_id':p.predicate_id,
         'comparison_group_attributes':[a.model_dump() for a in image_order_attributes(p)],
         'visible_only':p.predicate_id.rsplit(':',1)[-1].endswith('_visible'),
         'target_only_recognition_attributes':[a for a in payload['required_recognition_attributes']
             if a not in [v.model_dump() for v in image_order_attributes(p)]]}
        for p in facts if p.predicate_id.startswith('image_order:')
            and p.predicate_id in selected | plan_ids | option_ids]
    payload['grounded_writing_brief'] = {
        'perspective': 'person observing from the ego vehicle',
        'spatial_evidence': 'verified native 3D boxes and point support in the ego frame; calibrated image relations are explicitly camera-scoped',
        'appearance_evidence': 'confirmed RGB attributes of the same native unit and supplied landmarks',
        'writing_goal': 'one recognizable object, everyday English, only the useful distinguishing details',
        'cue_preference': ['confirmed specific type and salient appearance with ego-relative position',
                           'verified position relative to a recognizable landmark',
                           'explicit-camera horizontal order only if needed for uniqueness'],
        'landmark_rule': 'use only supplied independently identifiable landmarks and supplied relations',
        'generation_mode': 'model writes the expression; evidence rules are not sentence templates'}
    guide = SCENE_TASK_GUIDE if scene_reference_policy(packet) else TASK_GUIDE
    if scene_reference_policy(packet):
        payload['grounded_writing_brief']['cue_preference'] = [
            'confirmed specific type and salient intrinsic attributes',
            'supported ego-relative location', 'verified relation to a distinctive identifiable landmark']
        payload['grounded_writing_brief']['forbidden_cues'] = ['image ordering', 'camera visibility alone']
    if minimal_reference_policy(packet):
        guide = MINIMAL_TASK_GUIDE
        payload['required_core_type'] = None
        payload['grounded_writing_brief']['cue_preference'] = [
            'confirmed specific category and salient intrinsic properties with useful verified ego-relative position',
            'verified relation to a distinctive identifiable landmark',
            'explicit-camera horizontal reference only if scene-based options are insufficient',
            'original-image boundary truncation only as the final necessary disambiguator']
        payload['grounded_writing_brief'].pop('forbidden_cues',None)
    from app.quality.description_search import traffic_scene_policy
    if traffic_scene_policy(packet):
        guide = TRAFFIC_TASK_GUIDE
        payload['grounded_writing_brief']['cue_preference'] = [
            'confirmed specific category and salient intrinsic properties',
            'useful verified ego-relative position or independently identifiable landmark relation',
            'explicit-camera appearance or horizontal relation only when needed for uniqueness']
        payload['grounded_writing_brief']['forbidden_cues'] = ['image-edge truncation and camera framing']
        payload['grounded_writing_brief']['insufficient_evidence'] = (
            'obtain another fully verified unique scene reference or change view/frame; '
            'do not delete a necessary cue or replace image truncation with guessed proximity')
    from app.quality.spatial_relations import native_only
    if native_only(packet):
        guide = ('Write one concise natural English description identifying the same native 3D target. '
            'Return JSON with expression, predicate_ids and visual_attributes according to output_schema. '
            'Use only available_target_facts and confirmed RGB attributes. Every spatial relation must use '
            'native 3D coordinates, box extents or vehicle heading. NEVER use image above/below, image '
            'left/right, pixel ranks, image edges or apparent perspective as spatial evidence. '
            'Ego-relative directions follow the ego axes; vehicle front/behind follows the landmark vehicle heading. '
            'Preserve the supplied reference frame. Above/below requires real vertical separation and '
            'horizontal alignment, not a different vertical pixel position. '
            'Camera qualifiers scope appearance/visibility, not spatial coordinates. Unknown 3D identities '
            'cannot prove relations. Attributes or verified ego location may suffice: do not force a landmark. '
            'Choose a complete prechecked unique plan when available. Match declarations to ALL and ONLY '
            'spoken facts. Landmark attributes belong to the relation, not target visual_attributes. '
            'Do not invent motion, proximity, readable text, fine attributes, or IDs. Human feedback is not evidence.')
        payload['grounded_writing_brief'].update(
            spatial_evidence='native 3D box coordinates, extents and heading only; pixels establish appearance/identity',
            cue_preference=['confirmed attributes, with source view if needed','verified ego-relative location',
                'verified native 3D landmark relation only when it helps disambiguate'],
            forbidden_cues=['all pixel-based spatial relations and rankings','image-edge framing','unverified 3D identities'],
            landmark_rule='confirmed native identity and descriptor; the whole expression must identify one target')
    observed = (observations or {}).get(packet.target_object_id, {})
    retry_guide = ('\nThis is a manual-review retry. Address task.manual_review_feedback.reason and '
        'generate a new sentence for the SAME native target. Do not repeat the rejected sentence. '
        'Human feedback reports a problem; it is not verified visual evidence. '
        'Use only confirmed observations and geometry, and never invent attributes to satisfy feedback.'
        if packet.task.get('manual_review_feedback') else '')
    if (checks_rgb_visibility(packet)
            and packet.provenance.get('admission_plan', {}).get('passed')
            and usable_rgb_observation(observed, allow_partial=allows_partial(packet))):
        # Perception is already frozen and checked. Rendering its facts should
        # neither repeat image inference nor introduce new appearance guesses.
        payload['identity_checked_appearance_inputs'] = []
        payload['generation_evidence_source'] = 'confirmed RGB observations and verified geometry'
        source = next((v for v in packet.views if v.camera_label == packet.source_view), None)
        content = [{'type':'text', 'text':json.dumps(payload)}]
        if source and source.artifacts.get('scene'):
            name = source.artifacts['scene']
            photo = checked_asset(EVIDENCE_ROOT / packet.packet_id, name,
                packet.provenance['artifact_sha256'][name]).read_bytes()
            with Image.open(io.BytesIO(photo)) as decoded:
                display_width, display_height = decoded.size
            region = source.appearance_box_xyxy
            photo, mime, note = observation_image_input(photo)
            payload['source_target_region'] = {'original_xyxy':region,
                'original_size':[source.width,source.height],
                'context_xyxy':([region[0]*display_width/source.width,region[1]*display_height/source.height,
                    region[2]*display_width/source.width,region[3]*display_height/source.height] if region else None),
                'context_size':[display_width,display_height], 'identity_source':source.appearance_identity_source}
            content[0]['text'] = json.dumps(payload)
            content.extend([{'type':'text','text':'Unmarked source-view context, possibly resized. Original and context pixel coordinates are labeled separately. Use it for scene context and readable phrasing. '
                'Only the supplied confirmed attributes and verified spatial relations may become claims. ' + note},
                {'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(photo).decode()}}])
        return [{"role": "system", "content": guide + retry_guide +
            '\nRender only the supplied confirmed observations and verified facts. '
            'Appearance claims must come from the canonical attributes list, never from visibility_reason prose. '
            'The photograph is context, not permission to add unconfirmed attributes or geometry. '
            'Omit unconfirmed attributes; retain necessary distinguishing cues.'},
            {"role": "user", "content": content}]
    content = [{"type": "text", "text": json.dumps(payload)}]
    for view in packet.views[:2]:
        for kind in ["overlay", "crop", *(["appearance"] if view in appearance_views(packet) else [])]:
            name = view.artifacts[kind]
            image = checked_asset(EVIDENCE_ROOT / packet.packet_id, name, packet.provenance["artifact_sha256"][name]).read_bytes()
            with Image.open(io.BytesIO(image)) as decoded:
                width, height = decoded.size
            # A camera-edge sliver is retained as evidence, but is not a valid
            # provider input. Do not upscale it into apparently useful evidence.
            if min(width, height) < 10 or max(width, height) / min(width, height) > 200:
                content.append({"type": "text", "text": f"Annotation-only {view.camera_label} {kind} omitted: image dimensions {width}x{height} fail the provider input limits. Use the valid full-view overlay; this omitted crop supplies no evidence."})
                continue
            content.extend([{"type": "text", "text": (f"Identity-checked appearance crop {view.camera_label}; source: {view.appearance_identity_source}; use only clearly visible attributes." if kind == "appearance" else f"Annotation-only projection {view.camera_label} {kind}; no appearance identity proof. Camera image axes are not ego axes.")},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}}])
    return [{"role": "system", "content": guide + retry_guide}, {"role": "user", "content": content}]


def run_packet(packet: EvidencePacket, calls, max_repairs: int = 2, output_root: Path = RUN_ROOT,
               max_model_calls: int = 12) -> dict:
    """Compatibility adapter to the single Agent state machine, not a second engine."""
    from app.agent.agent_runtime import run_agent
    return run_agent(packet, calls, max_repairs, output_root, max_model_calls)
