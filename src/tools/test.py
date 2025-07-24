
import numpy as np
import geopandas as gpd
import os
from os.path import join
import json
from PIL import Image

from src.data.dataset_tiler import TileSlidingGeneratorShapes
from src.data.dataset_cropper import OrthoCropDataset

from src.data.utils_api import OrthoCropApi

"""
python -m src.tools.test \
      --name test_it \
      --modality BDORTHO \
      --path_to_logs /home/adhemar/Bureau/METHAN/code/logs/train_bdortho_150cm_01 \
      --path_to_test /media/adhemar/Disc4T1/METHAN/V_multy_source_2/res_1.5/test \
      --get_map \
      --get_ap_dist

python -m src.tools.test \
      --name test_it \
      --modality SPOT \
      --path_to_logs /home/adhemar/Bureau/METHAN/code/logs/train_spot_150cm_01 \
      --path_to_test /media/adhemar/Disc4T1/METHAN/V_multy_source_2/res_1.5/test \
      --get_map \
      --get_ap_dist


python -m src.tools.test \
      --name test_it \
      --modality BDORTHO \
      --path_to_logs /home/adhemar/Bureau/METHAN/public/logs/train_bdortho_150cm_01 \
      --path_to_test /media/adhemar/Disc4T1/METHAN/V_multy_source_2/res_1.5/test \
      --get_map \
      --get_ap_dist

python -m src.tools.test \
      --name test_it \
      --modality SPOT \
      --path_to_logs /home/adhemar/Bureau/METHAN/public/logs/train_spot_150cm_01 \
      --path_to_test /media/adhemar/Disc4T1/METHAN/V_multy_source_2/res_1.5/test \
      --get_map \
      --get_ap_dist
"""

def precision_recall_all(gt_positions, pred_positions, threshold=200):
    """
    Compute precision & recall when accepting all predictions.
    """
    # match preds → gts
    gt_matched = np.zeros(len(gt_positions), dtype=bool)
    tp = 0
    for pos in pred_positions:
        dists = np.linalg.norm(np.array(gt_positions) - pos, axis=1)
        i = np.argmin(dists)
        if dists[i] <= threshold and not gt_matched[i]:
            tp += 1
            gt_matched[i] = True
    fp = len(pred_positions) - tp
    fn = len(gt_positions) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    print(f"P@all: {precision:.4f}, R@all: {recall:.4f}")
    return precision, recall

def test_ap(gt_positions, pred_positions, pred_scores, threshold=200):
    """
    Compute Average Precision (AP) for detections based on a distance threshold.
    
    Args:
        gt_positions (list of tuple): Ground‑truth centers [(x, y), ...].
        pred_positions (list of tuple): Predicted centers [(x, y), ...].
        pred_scores (list of float): Confidence scores for each prediction.
        threshold (float): Distance threshold for true positives (in meters).
        
    Returns:
        float: AP value.
    """
    # Sort predictions by descending score
    order = np.argsort(pred_scores)[::-1]
    pred_positions = [pred_positions[i] for i in order]
    matches = []
    gt_matched = np.zeros(len(gt_positions), dtype=bool)
    
    for pos in pred_positions:
        dists = np.linalg.norm(np.array(gt_positions) - pos, axis=1)
        idx_min = np.argmin(dists)
        if dists[idx_min] <= threshold and not gt_matched[idx_min]:
            matches.append(1)
            gt_matched[idx_min] = True
        else:
            matches.append(0)
    
    tp = np.cumsum(matches)
    fp = np.cumsum([1 - m for m in matches])
    recalls = tp / len(gt_positions)
    precisions = tp / (tp + fp)
    
    # Compute AP via precision-recall curve area
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    
    print(f"AP @ {threshold}m: {ap:.4f}")
    return ap

def find_unique_py_file(path_to_logs):
    py_files = [f for f in os.listdir(path_to_logs) if f.endswith('.py')]
    if len(py_files) == 0:
        raise FileNotFoundError("No .py file found in the directory.")
    if len(py_files) > 1:
        raise RuntimeError(f"Multiple .py files found: {py_files}")
    return os.path.join(path_to_logs, py_files[0])

class OrthoCropApiTestEnv(OrthoCropApi):
    def __init__(
            self,
            path_to_tile_json,
            path_to_images,
            **kwargs,
        ):

        self.path_to_images = path_to_images
        self.epsg_query = "EPSG:2154"

        with open(path_to_tile_json, "r") as fp:
            self.tiles = json.load(fp)
        
        super().__init__(**kwargs)
        

    def get_once(self, bbox, *args, **kwargs):
        match = next((t for t in self.tiles if t['bbox_epsg_image'] == bbox), None)
        
        filename = match['filename']
        img_path = os.path.join(self.path_to_images, filename)
        with Image.open(img_path) as img:
            return {"image": img.copy()}
    
# -- MAIN -- #

def main(
        name,
        modality, # SPOT or BDORTHO
        path_to_logs,
        path_to_test,
        infer = True,
        get_map = True,
        use_last = False,
        get_ap_dist = False,
):
    path_to_exp = os.path.join(path_to_logs, 'test', name)



    # -- DATASET -- #
    api = OrthoCropApiTestEnv(
        path_to_tile_json = os.path.join(path_to_test, 'meta', 'tiles.json'),
        path_to_images = os.path.join(path_to_test, 'image', modality),
        
    )


    DATASET = OrthoCropDataset(
        api=api,
        resolution=1.5,
        window_size=1536,
        epsg_in="EPSG:2154",
    )

    # -- GENERATOR -- #
    gdf_visu_region = gpd.read_file(os.path.join(path_to_test, "meta/gadm41_FRA_2.shp"))
    region = gdf_visu_region[gdf_visu_region["NAME_1"] == "Grand Est"]
    region = region.to_crs("EPSG:2154")
    marne = region[region['NAME_2'] == "Marne"]

    GENERATOR = TileSlidingGeneratorShapes(
        zones_to_map = marne,
        zones_of_tiles = marne,
        epsg = 'EPSG:2154',
        folder = '/media/adhemar/Disc4T1/METHAN/V_multy_source_2/res_1.5/test/meta',
        tiling_config = {
            'tile_size_metre': 1536,
            'hop_fraction': 0.85,
        },
        signed_distance_treshold_margine = 20000,
        folder_results = path_to_exp

    )

    GENERATOR.construct_tiling()

    
    if infer:
        from src.utils.model_warper_mmrotate import ModelWarperTest
        path_to_config = find_unique_py_file(path_to_logs)
        if use_last:
            path_to_weights = os.path.join(path_to_logs, 'model_last.pth')
        else:
            path_to_weights = os.path.join(path_to_logs, 'model_best.pth')

        
        MODEL = ModelWarperTest(
            path_to_config, 
            path_to_weights,
        )
        
        # run:
        GENERATOR.run(
            DATASET,
            MODEL,
            time_between_saves = 1*20*60,
            resume=True, # Resum last run
        )

 
        
    else:
        pass

    
    post_process_config = {
        "threshold": 0.000096, # 0.000096
        "methode": 'proba_histogram',
        "pre_biodigestor_threshold": 0.05,
        "skip": False,
    }

    GENERATOR.post_process(post_process_config)

    
    if get_map:
        
        GENERATOR.get_map(
            add_raw_detections = False, 
            add_selected_detections = True
        ) # add pre filtering ! important

    if get_ap_dist:
        
        selected_detections =  GENERATOR._load_selected_detections()


        path_to_test_positions = join(path_to_test, 'label', modality, 'position.json')
        with open(path_to_test_positions, 'r') as f:
            position_gt = json.load(f)
        
        print("Total number of positions: ", len(position_gt), "(Ground Truth)")
        print("Total number of detections: ", len(selected_detections), "(Detections)")
        
        positions_found, scores = [], []
        for det in selected_detections:
            x, y, w, h = det["bbox"]
            centre = (x + w/2, y + h/2)
            positions_found.append(centre)
            scores.append(det["global_score"])


        ap    = test_ap(position_gt, positions_found, scores)
        prec, rec = precision_recall_all(position_gt, positions_found)

        # pack and save
        metrics = {'ap': ap, 'precision': prec, 'recall': rec}
        out_path = os.path.join(path_to_exp, 'metrics.npy')
        np.save(out_path, metrics)  # saved as a pickled dict in .npy
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)

