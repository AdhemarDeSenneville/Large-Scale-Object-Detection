
from .val_github import val, generate_results, eval_coco
from .val_detection import eval_ap
import os
import fire


"""
python -m src.tools.eval \
  --name val_test \
  --path_to_logs /home/adhemar/Bureau/METHAN/code/logs/train_spot_050cm_01 \
  --path_to_imgs /home/adhemar/Bureau/datasets/Methanizers/V_multy_source/res_0.5/image/SPOT/val \
  --path_to_json /home/adhemar/Bureau/datasets/Methanizers/V_multy_source/res_0.5/label/val/annotations.json \
  --vpv --infer False
  
python -m src.tools.eval \
  --name val_test \
  --path_to_logs /home/adhemar/Bureau/METHAN/public/logs/train_spot_150cm_01 \
  --path_to_imgs /media/adhemar/Disc4T1/METHAN/V_multy_source_2/res_1.5/image/SPOT/val \
  --path_to_json /media/adhemar/Disc4T1/METHAN/V_multy_source_2/res_1.5/label/val/annotations.json \
  --vpv --infer
"""

def find_unique_py_file(path_to_logs):
    py_files = [f for f in os.listdir(path_to_logs) if f.endswith('.py')]
    if len(py_files) == 0:
        raise FileNotFoundError("No .py file found in the directory.")
    if len(py_files) > 1:
        raise RuntimeError(f"Multiple .py files found: {py_files}")
    return os.path.join(path_to_logs, py_files[0])

def main(
        name,
        path_to_logs,
        path_to_imgs,
        path_to_json,
        vpv = True,
        infer = True,
        get_coco = True,
        get_ap = True,
        use_last = False,
):
    path_to_exp = os.path.join(path_to_logs, 'eval', name)
    
    if infer:
        from src.utils.model_warper_mmrotate import ModelWarper
        path_to_config = find_unique_py_file(path_to_logs)
        if use_last:
            path_to_weights = os.path.join(path_to_logs, 'model_last.pth')
        else:
            path_to_weights = os.path.join(path_to_logs, 'model_best.pth')

        
        model = ModelWarper(
            path_to_config, 
            path_to_weights,
        )


        # Val Hard
        path_to_vpv, path_to_model_outputs = generate_results(
            model,
            path_to_exp ,
            path_to_imgs, 
            path_to_json,
            path_to_vpv = vpv,
            score_threshold = 0,
        )
    else:
        path_to_model_outputs = os.path.join(path_to_exp, 'model_outputs.json')
        path_to_vpv = os.path.join(path_to_exp, 'vpv_outs')
    
    if vpv:
        vpv_command = (
            f"vpv ac aw "
            f"nw {path_to_imgs}/ "
            f"svg:{path_to_vpv}/ "
        )
        print()
        print(vpv_command)
        print()

    if get_coco:
        eval_coco(
            path_to_exp,
            path_to_json,
            path_to_model_outputs,
        )
    
    if get_ap:
        eval_ap(
            path_to_exp,
            path_to_json,
            path_to_model_outputs,
        )



if __name__ == '__main__':
    fire.Fire(main)