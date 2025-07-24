from collections import Counter

import pandas as pd
import numpy as np
from collections import defaultdict
import json

def json_to_geojson(
        features, 
        out_path,
        name = "save_geo_json"
    ):
    geojson = {
        "type": "FeatureCollection",
        "name": name,
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::2154" } },
        "features": features
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=4)
    
    print(f'Saved GEOJSON {name} to {out_path}')


def geojson_to_json(path):
    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    
    print(f'Loaded GEOJSON from {path}')
    return geojson['features']

class GeoStatDatabase:
    def __init__(
            self, 
            path = '/home/adhemar/Bureau/datasets/Osint/DatabaseFull_V2_130725.geojson', 
            base_crs=2154
        ):
        self.data = geojson_to_json(path)
        self.crs = base_crs
        self.sources = self.get_all_sources()

    def get_all_sources(self):
        det, ann, osi = set(), set(), set()
        for feat in self.data:
            props = feat['properties']
            det |= set(props.get('detection', {}))
            ann |= set(props.get('annotation', {}))
            osi |= set(props.get('osint', {}))
        print("Detections:", det)
        print("Annotations:", ann)
        print("OSINT:", osi)

        self.detection_names = det
        self.annotation_names = ann
        self.osint_names = osi
        
        return {'detection': det, 'annotation': ann, 'osint': osi}

    def annotation_info(self, annotation_name):
        counts = Counter()
        for feat in self.data:
            ann = feat['properties'].get('annotation', {})
            info = ann.get(annotation_name)
            if info:
                lbl = info['data'].get('label')
                if lbl is not None:
                    counts[lbl] += 1
        total = sum(counts.values())
        n = len(self.data)
        print(f"== {annotation_name} ==")
        print(f"Total annotations: {total}/{n}")
        for val, cnt in counts.items():
            print(f"  {val}: {cnt}")

    def all_annotations_info(self):
        for name in self.sources['annotation']:
            self.annotation_info(name)
        
    def get_source(self, idx, source_name):
        # return none if not found
        # check if in detection, annotation or osint
        # return the source
        feat = self[idx]
        props = feat['properties']
        if source_name in props.get('detection', {}):
            return props['detection'][source_name]
        elif source_name in props.get('annotation', {}):
            return props['annotation'][source_name]
        elif source_name in props.get('osint', {}):
            return props['osint'][source_name]
        else:
            return None
    
    def get_idxs(self, source_name):
        """
        Return list of feature indices where `source_name` appears
        in detection, annotation or osint.
        """
        idxs = []
        for idx, feat in enumerate(self.data):
            props = feat['properties']
            if (source_name in props.get('detection', {})
                or source_name in props.get('annotation', {})
                or source_name in props.get('osint', {})):
                idxs.append(idx)
        return idxs    
    
    def __getitem__(self, idx):
        """
        Allow indexing like db[23] to return the feature at position 23.
        """
        return self.data[idx]
    
    def show_key_tree(self, idx):
        """
        Print a tree of all keys (no values) in feature at index `idx`.
        """
        def recurse(obj, level=0):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    print('  ' * level + k)
                    recurse(v, level + 1)
            elif isinstance(obj, list):
                for item in obj:
                    recurse(item, level)
        feature = self[idx]  # uses __getitem__
        recurse(feature)


import matplotlib.pyplot as plt
def show_source_overlap_matrix(db, filter_label=None, label_key=None):
    records = []
    for idx, feat in enumerate(db.data):
        if filter_label and label_key:
            ann = feat['properties'].get('annotation', {})
            lbl = ann.get(label_key, {}).get('data', {}).get('human_feedback', {}).get(filter_label)
            if lbl != 'True':
                continue
        for cat in ('annotation', 'detection', 'osint'):  # Changed order: annotation before detection
            for src in getattr(db, f"{cat}_names"):
                if src in feat['properties'].get(cat, {}):
                    records.append({'_idx': idx, 'category': cat, 'source': src})
    bin_df = pd.DataFrame(records)
    indicator = (bin_df.assign(present=1)
                   .pivot_table(index='_idx', columns='source', values='present', fill_value=0))
    order = sorted(db.annotation_names) + sorted(db.detection_names) + sorted(db.osint_names)  # Changed order
    indicator = indicator.reindex(columns=order, fill_value=0)
    
    joint = indicator.T.dot(indicator)
    marginals = indicator.sum(axis=0)
    pct = joint.div(marginals, axis=0).mul(100).round(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                   gridspec_kw={'width_ratios': [4, 1]})
    im = ax1.imshow(pct, origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax1, label='% overlap')
    ax1.set_xticks(range(len(order))); ax1.set_xticklabels(order, rotation=90)
    ax1.set_yticks(range(len(order))); ax1.set_yticklabels(order)
    ax1.set_title('Directional overlap: P(source_j | source_i) ×100')

    colors = {'detection': 'C0', 'annotation': 'C1', 'osint': 'C2'}
    cats = ['annotation' if src in db.annotation_names 
            else 'detection' if src in db.detection_names 
            else 'osint' for src in order]  # Changed order
    ax2.barh(range(len(order)), marginals[order], color=[colors[c] for c in cats])
    ax2.set_yticks(range(len(order))); ax2.set_yticklabels(order)
    ax2.set_title('Marginal counts')

    # Add black lines to separate categories in the matrix
    annotation_end = len(db.annotation_names)  # Changed order
    detection_end = annotation_end + len(db.detection_names)  # Changed order

    ax1.axhline(annotation_end - 0.5, color='black', linewidth=1.5)  # Changed order
    ax1.axhline(detection_end - 0.5, color='black', linewidth=1.5)  # Changed order
    ax1.axvline(annotation_end - 0.5, color='black', linewidth=1.5)  # Changed order
    ax1.axvline(detection_end - 0.5, color='black', linewidth=1.5)  # Changed order

    # Add percentage numbers on each square
    for i in range(len(order)):
        for j in range(len(order)):
            value = pct.iloc[i, j]
            ax1.text(j, i, f"{value:.0f}%", ha='center', va='center', fontsize=8, color='black')

    ax1.set_xticklabels(order, rotation=45, ha='right')
    ax1.set_yticklabels(order, rotation=45, va='top')
    ax2.set_yticklabels(order, rotation=45, va='top')
    
    plt.tight_layout()
    plt.show()
