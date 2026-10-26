"""Continuation accounting survives removal of an ancestor from the task list."""
import json


def read_lineage(db):
    live = {row['job_id']: {**dict(row), 'spec': json.loads(row['spec'])}
            for row in db.execute('SELECT * FROM batches')}
    retained, parents = {}, {}
    for row in db.execute("SELECT job_id,kind,payload FROM batch_events WHERE kind IN ('continued_from','retained_task_history') ORDER BY seq"):
        payload = json.loads(row['payload'])
        if row['kind'] == 'continued_from':
            parents[row['job_id']] = payload['job_id']
        else:
            for job, record in payload['tasks'].items():
                if job in retained and retained[job] != record:
                    raise ValueError('Conflicting retained task history')
                retained[job] = record
    for job, record in retained.items():
        if record.get('parent_job_id'):
            parents[job] = record['parent_job_id']
    return live, retained, parents


def lineage_family(job, parents):
    root, seen = job, set()
    while root in parents:
        if root in seen:
            raise ValueError('Invalid continuation cycle')
        seen.add(root)
        root = parents[root]
    family = {root}
    while True:
        expanded = family | {child for child, parent in parents.items() if parent in family}
        if expanded == family:
            return root, family
        family = expanded
