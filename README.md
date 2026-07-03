# Towards Large Scale Geostatistical Methane Monitoring with Part‑based Object Detection

[Adhémar de Senneville](https://adhemardesenneville.github.io/) · [Xavi Bou](https://xavibou.github.io/) · [Thibaud Ehret](https://scholar.google.fr/citations?user=nnCC19cAAAAJ&hl=en) · [Rafael Grompone](https://centreborelli.ens-paris-saclay.fr/fr/annuaire-des-personnes/raffaele-grompone) · [Nicolas Dumelie](https://cv.hal.science/nicolas-dumelie) · [Jean‑Louis Bonne](https://scholar.google.com/citations?user=HcKn7BIAAAAJ&hl=fr) · [Thomas Lauvaux](https://www.cefe.cnrs.fr/fr/recherche/ef/forecast/832-v/3657-lauvaux-thomas) · [Gabriele Facciolo](http://gfacciol.github.io/)

Centre Borelli, ENS Paris‑Saclay & Université de Reims

![Overview of our results in the French Grand Est region with the number of detected bio-digester sites in each department in 2023. We use our model to detect unknown bio-digester sites in large areas. On the right, we show (a) some examples of annotated bio-digester sites (from the validation set) with their sub-elements. (b) Shows predictions from our model; even with detection errors and a small training set, the part-based detector reliably identifies bio-digester sites at scale.](assets/main.svg)

## Resources

- 📄 **Website**: [github.io](https://adhemardesenneville.github.io/Large-Scale-Object-Detection/)
- 📄 **Paper**: [arXiv 2304.06871](https://arxiv.org/abs/2507.18513)
- 📦 **Dataset**: [Zenodo Record](https://zenodo.org/records/16411300)  

## **Overview**

This repository provides tools for detecting **methane digesters** in large-scale **RGB satellite imagery** using **part-based object detection** and for estimating geostatistical methane emissions.
Our approach leverages **MMRotate** for rotated object detection, enabling robust detection across various resolutions and imaging modalities (SPOT, BDORTHO).


---

## **Installation**

```bash
git clone https://github.com/AdhemarDeSenneville/Large-Scale-Object-Detection.git
cd Large-Scale-Object-Detection
pip install -r requirements.txt
```

**Requirements**:

* Python 3.8+
* PyTorch 1.8+
* CUDA 11+
* [MMRotate](https://github.com/open-mmlab/mmrotate)

---

## **Evaluation**

```bash
python -m src.tools.eval \
  --name val_test \
  --path_to_logs /path/to/logs/train_spot_050cm_01 \
  --path_to_imgs /path/to/images/SPOT/val \
  --path_to_json /path/to/val/annotations.json \
  --vpv --infer False
```

| Argument         | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| `--name`         | Name of the evaluation run.                                         |
| `--path_to_logs` | Directory mmrotate with model checkpoints (.pth) and config (.py).  |
| `--path_to_imgs` | Path to validation images.                                          |
| `--path_to_json` | Path to COCO-format annotations file.                               |
| `--vpv`          | Generates visualizations of predictions vs ground truth.            |
| `--infer`        | `True` → runs inference again, `False` → uses existing predictions. |

---

## **Testing**

Run evaluation on an **independent test set** to compute **mAP** and analyze performance distribution.

### **SPOT Example**

```bash
python -m src.tools.test \
  --name test_it \
  --modality SPOT \
  --path_to_logs /path/to/logs/train_spot_150cm_01 \
  --path_to_test /path/to/test/images \
  --get_map \
  --get_ap_dist
```

### **BDORTHO Example**

```bash
python -m src.tools.test \
  --name test_it \
  --modality BDORTHO \
  --path_to_logs /path/to/logs/train_bdortho_150cm_01 \
  --path_to_test /path/to/test/images \
  --get_map \
  --get_ap_dist
```
 
| Argument         | Description                                                          |
| ---------------- | -------------------------------------------------------------------- |
| `--name`         | Name of the test run.                                                |
| `--modality`     | Satellite image source                                               |
| `--path_to_logs` | Directory mmrotate with model checkpoints (.pth) and config (.py).   |
| `--path_to_test` | Path to test dataset.                                                |
| `--get_map`      | Export an html map of detections.                                    |
| `--get_ap_dist`  | Computes mean Average Precision at 200m (mAP).                       |

---

## **Model Zoo**

| Model         | Resolution | Config File                      | Checkpoint   |
| ------------- | ---------- | -------------------------------- | ------------ |
| SPOT-150cm    | 1.5 m      | `configs/train_spot_150cm`       | [Download]() |
| BDORTHO-150cm | 1.5 m      | `configs/train_bdortho_150cm`    | [Download]() |

---

## **Training**

Training is based on **MMRotate**. All configurations are available in the `configs/` folder.
Modify the configuration file to adapt backbone, data resolution, or augmentation strategies.

---

## **Citation**

If you use this work, please cite:

```bibtex
@InProceedings{de_Senneville_2025_ICCV,
    author    = {de Senneville, Adhemar and Bou, Xavier and Bonne, Jean-Louis and Dumelie, Nicolas and Grompone, Rafael and Ehret, Thibaud and Facciolo, Gabriele},
    title     = {Towards Large Scale Geostatistical Methane Monitoring with Part-based Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {2886-2896}
}
```
