"""Build official same-ID projection candidates; optional local frame preparation."""
import argparse
import json

from app.datasets.catalog import build_catalog
from app.datasets.evidence import ROOT
from app.datasets.waymo import WaymoReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-frames',type=int,default=0,choices=range(0,11),
        help='prepare up to N additional candidate frames locally')
    args = parser.parse_args()
    catalog = build_catalog(WaymoReader(ROOT / "waymo_trainset"))
    limit = args.prepare_frames
    if limit:
        from app.datasets.preparation import ConversionSession
        index_path = ROOT/'runtime/scenes/index.json'
        index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {'scenes':[]}
        seen = {(s['context'],s['timestamp']) for s in index['scenes']}
        candidates = sorted([t for t in catalog['targets'] if t['status']=='selected'],
            key=lambda t:t.get('recommendation_rank', t['ordinal']))
        converter = ConversionSession()
        try:
            prepared = 0
            for target in candidates:
                key = (target['context'],target['timestamp'])
                if key in seen:
                    continue
                print('Preparing candidate frame',*key,flush=True)
                last_stage = None
                def progress(stage=None):
                    nonlocal last_stage
                    if stage and stage != last_stage:
                        last_stage = stage
                        print(stage,flush=True)
                converter(*key,progress,ROOT/'runtime/image_pairings/bootstrap'/str(target['timestamp'])/'preparation.log')
                seen.add(key); prepared += 1
                if prepared >= limit:
                    break
        finally:
            converter.close()
        catalog = build_catalog(WaymoReader(ROOT/'waymo_trainset'))
    print(catalog["catalog_id"], catalog["summary"], flush=True)


if __name__ == "__main__":
    main()
