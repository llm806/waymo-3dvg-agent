"""Versioned RGB admission: recognizability is separate from completeness."""
RECOGNIZABLE_POLICY = 'recognizable-source-rgb-v2'
SUPPORTED_POLICIES = {'clear-source-rgb-v1', RECOGNIZABLE_POLICY}


def allows_partial(packet):
    return packet.task.get('visibility_policy') == RECOGNIZABLE_POLICY


def checks_rgb_visibility(packet):
    return packet.task.get('visibility_policy') in SUPPORTED_POLICIES


def usable_rgb_observation(record, *, allow_partial=False):
    accepted = {'clear', 'partial'} if allow_partial else {'clear'}
    return (record.get('target_visible') is True and record.get('visibility') in accepted
        and bool(record.get('visibility_reason', '').strip()))


def visibility_evidence(record, *, allow_partial=False):
    """Unknown recognition is not evidence that the object is fully hidden."""
    if usable_rgb_observation(record, allow_partial=allow_partial):
        return True
    if (record.get('target_visible') is False and record.get('visibility') == 'occluded'
            and record.get('visibility_reason', '').strip()):
        return False
    return None


def source_camera_inventory(packet):
    key = 'recognizable_source_cameras' if allows_partial(packet) else 'unobstructed_source_cameras'
    return packet.provenance.get(key)
