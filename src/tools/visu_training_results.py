
from utils_visu import mmrotate_get_logs, mmrotate_clear_checkpoints, plot_metrics_vs_steps
from os.path import join
import os
import fire

def main(
        path_to_log,
        verbose = False,
        backend = 'mmrotate',
        clear_ckpt = False,
):
    path_to_save = join(path_to_log, 'visu')
    os.makedirs(path_to_save, exist_ok=True)
    if backend == 'mmrotate':

        train_logs, val_logs = mmrotate_get_logs(path_to_log, verbose = False)

        if clear_ckpt:
            mmrotate_clear_checkpoints(path_to_log, val_logs)

        plot_metrics_vs_steps(
            val_logs, 
            keys = 'mAP', 
            steps_per_epoch = 150,
            save_path = join(path_to_save, 'visu_val_map.pdf'),
            verbose = verbose,
        )
        plot_metrics_vs_steps(
            train_logs, 
            keys = ["loss", "loss_cls", 'loss_rpn_cls', 'loss_rpn_bbox'],
            steps_per_epoch = 150,
            save_path = join(path_to_save, 'visu_train_loss.pdf'),
            verbose = verbose,
        )
        plot_metrics_vs_steps(
            train_logs, 
            keys = "lr", 
            steps_per_epoch = 150,
            save_path = join(path_to_save, 'visu_train_lr.pdf'),
            verbose = verbose,
        )

    else:
        raise NotImplementedError


if __name__ == '__main__':
    fire.Fire(main)