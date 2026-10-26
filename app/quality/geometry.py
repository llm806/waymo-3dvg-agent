"""Numerical geometry shared across incompatible SDK and annotation environments."""

import numpy as np
import math


def native_box_proximity(target, other, *, planning=False):
    """Flag nearly coincident same-class units for planning/diagnosis only.

    This is not an identity join, IoU certificate or reason to remove a box.
    """
    if (target.object_id == other.object_id or target.category != other.category
            or not target.geometry_valid or not other.geometry_valid):
        return None
    gap = math.dist(target.center_m, other.center_m)
    if any(min(a,b) <= 0 for a,b in zip(target.size_m, other.size_m)):
        return None
    ratio = min(min(a,b)/max(a,b) for a,b in zip(target.size_m, other.size_m))
    angle = abs((target.heading_rad-other.heading_rad+math.pi/2) % math.pi-math.pi/2)
    if ratio < .8:
        return None
    if planning:
        # A broad source-selection risk: centers lie within half the smaller
        # extent along BOTH boxes' axes. Heading differences do not make two
        # overlapping people independently visible. This never rejects a frame.
        dx,dy,dz = (a-b for a,b in zip(target.center_m,other.center_m))
        half = [min(a,b)/2 for a,b in zip(target.size_m,other.size_m)]
        for heading in (target.heading_rad,other.heading_rad):
            c,s = math.cos(heading),math.sin(heading)
            if any(abs(v)>limit for v,limit in zip((dx*c+dy*s,-dx*s+dy*c,dz),half)):
                return None
    elif gap > .2 or angle > .35:
        return None
    return {'center_gap_m':gap,'minimum_size_ratio':ratio}


def inside_oriented_box(xyz, center_m, size_m, heading_rad, tolerance_m=0.01):
    xyz = np.asarray(xyz)
    center, size = np.asarray(center_m), np.asarray(size_m)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Finite Nx3 coordinates required")
    if center.shape != (3,) or size.shape != (3,) or not np.isfinite([*center, *size, heading_rad]).all() or np.any(size <= 0):
        raise ValueError("Finite native 7-DOF box with positive sizes required")
    if not 0 <= tolerance_m <= 0.1:
        raise ValueError("Box tolerance must be 0..0.1 meters")
    delta = xyz - center
    cosine, sine = np.cos(heading_rad), np.sin(heading_rad)
    half = size / 2 + tolerance_m
    # Identical arithmetic and inclusive boundaries, without allocating and
    # reducing two extra Nx3 arrays for every native object in the frame.
    return ((np.abs(delta[:, 0] * cosine + delta[:, 1] * sine) <= half[0])
        & (np.abs(-delta[:, 0] * sine + delta[:, 1] * cosine) <= half[1])
        & (np.abs(delta[:, 2]) <= half[2]))


class BoxPointIndex:
    """Frame-local broad phase; final membership uses the exact original rule.

    Sort coordinates once instead of rotating the entire cloud for every box.
    Nothing is sampled, moved or normalized. Callers must keep this frame's
    array unchanged for the lifetime of the index; never reuse across frames.
    """
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not np.isfinite(self.xyz).all():
            raise ValueError("Finite Nx3 coordinates required")
        self.axes = []
        if len(self.xyz) >= 2048:
            for axis in range(3):
                # Promote float32 keys before search so a double-precision
                # bound cannot be rounded inward to a float32 search key.
                values = self.xyz[:, axis].astype(np.result_type(self.xyz.dtype, np.float64))
                order = np.argsort(values)
                self.axes.append((values[order], order))

    def inside(self, center_m, size_m, heading_rad, tolerance_m=0.01):
        if not self.axes:
            return inside_oriented_box(self.xyz, center_m, size_m, heading_rad, tolerance_m)
        # Validate the entire box contract even when the broad phase is empty.
        inside_oriented_box(self.xyz[:0], center_m, size_m, heading_rad, tolerance_m)
        center, half = np.asarray(center_m), np.asarray(size_m) / 2 + tolerance_m
        # A deliberately loose square encloses every heading. Numerical slack
        # plus outward rounding protects inclusive faces and thin native parts.
        radii = (2 * (half[0] + half[1]), 2 * (half[0] + half[1]), 2 * half[2])
        choices = []
        for axis, (values, order) in enumerate(self.axes):
            low = np.nextafter(center[axis] - radii[axis], -np.inf)
            high = np.nextafter(center[axis] + radii[axis], np.inf)
            start, stop = np.searchsorted(values, low, side='left'), np.searchsorted(values, high, side='right')
            choices.append((stop-start, axis, start, stop))
        _, axis, start, stop = min(choices)
        indices = self.axes[axis][1][start:stop]
        mask = np.zeros(len(self.xyz), dtype=bool)
        mask[indices] = inside_oriented_box(self.xyz[indices], center_m, size_m, heading_rad, tolerance_m)
        return mask
