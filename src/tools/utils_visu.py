

import uuid
from IPython.display import display, HTML
import os
import json
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns; sns.set()
except:
    print('Impossible to import seaborn')

def plot_metrics_vs_epoch(formatted_logs, keys):
    # If keys is a single string, convert it to a list
    if isinstance(keys, str):
        keys = [keys]

    # Make sure "epoch" exists in the logs
    if "epoch" not in formatted_logs:
        raise ValueError("`formatted_logs` must contain an 'epoch' field.")

    # Identify unique epochs
    unique_epochs = sorted(set(formatted_logs["epoch"]))

    # Create a dictionary of indices for each epoch
    epoch_indices = {epoch: [] for epoch in unique_epochs}
    for i, epoch in enumerate(formatted_logs["epoch"]):
        epoch_indices[epoch].append(i)

    # For each metric, compute the mean per epoch
    mean_metrics_per_epoch = {key: [] for key in keys}
    for epoch in unique_epochs:
        indices_for_epoch = epoch_indices[epoch]
        for key in keys:
            if key not in formatted_logs:
                continue
            values_this_epoch = [formatted_logs[key][idx] for idx in indices_for_epoch]
            mean_value = sum(values_this_epoch) / len(values_this_epoch)
            mean_metrics_per_epoch[key].append(mean_value)

    # Plot
    plt.figure(figsize=(15, 8))
    for key in keys:
        # Plot the average of each metric vs. epoch
        plt.plot(
            unique_epochs,
            mean_metrics_per_epoch[key],
            marker='o',
            linestyle='-',
            label=f'{key.capitalize()}'
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Metrics as a Function of Epoch", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

def plot_metrics_vs_steps(formatted_logs, keys, steps_per_epoch = 1016, save_path = None, verbose = True):
    
    if isinstance(keys, str):
        keys = [keys]
    # Compute steps from epoch and iteration
    if "epoch" in formatted_logs and "iter" in formatted_logs:
        steps = [(epoch - 1) * steps_per_epoch + step for epoch, step in zip(formatted_logs["epoch"], formatted_logs["iter"])]
    else:
        steps = [i*steps_per_epoch for i in range(len(formatted_logs[keys[0]]))]

    # Plot each metric
    plt.figure(figsize=(15, 8))
    for key in keys:
        if key in formatted_logs:
            metric_values = formatted_logs[key]
            plt.plot(steps, metric_values, marker='o', linestyle='-', label=f'{key.capitalize()}')

    # Add labels, title, legend, and grid
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Metrics as a Function of Steps", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)

    if save_path is not None:
        plt.savefig(
            save_path
        )

    # Display the plot
    if verbose:
        plt.show()
    else:
        plt.close()




def display_command(command, title = 'COMMAND', *files):
    # Generate a unique function name
    unique_id = uuid.uuid4().hex  # or just use something like str(random.randint(0, 999999))

    button_html = f"""
    <script>
    function copyToClipboard_{unique_id}() {{
        const text = `{command}`;
        navigator.clipboard.writeText(text).then(
            () => alert('Command copied to clipboard!'),
            () => alert('Failed to copy command.')
        );
    }}
    </script>

    <button onclick="copyToClipboard_{unique_id}()">Copy Command</button>
    """

    print(title.center(100, '='))

    for file in files:
        print('From file:', file)

    print(command)
    display(HTML(button_html))
    print(100*'_')


# ---------------------- #
#                        #
#     mmrotate utils     #
#                        #
# ---------------------- #

def mmrotate_get_logs(path_logging_dir, verbose = False):
    
    # Get and sort JSON files by modification time
    json_files = sorted(
        [f for f in os.listdir(path_logging_dir) if f.endswith('.json')],
        key=lambda x: os.path.getmtime(os.path.join(path_logging_dir, x))
    )

    all_train_logs = []
    all_val_logs = []
    eval_config = None

    for json_file in json_files:
        if verbose: print(f"Processing: {json_file}")
        
        with open(os.path.join(path_logging_dir, json_file), 'r') as file:
            logs = [json.loads(line) for line in file]

            # Optionally capture the config from the first log file
            if eval_config is None and logs:
                eval_config = logs[0].get('config', None)

            # Filter relevant logs
            relevant_logs = [log for log in logs if "mode" in log and "epoch" in log]

            all_train_logs.extend([log for log in relevant_logs if log["mode"] == "train"])
            all_val_logs.extend([log for log in relevant_logs if log["mode"] == "val"])

    # Convert list of logs to dict of lists
    if all_train_logs:
        train_logs = {key: [log[key] for log in all_train_logs] for key in all_train_logs[0]}
    else:
        train_logs = {}

    if all_val_logs:
        val_logs = {key: [log[key] for log in all_val_logs] for key in all_val_logs[0]}
    else:
        val_logs = {}

    if verbose:
        print('Total train logs:', len(all_train_logs))
        print('Total val logs:', len(all_val_logs))
        if eval_config:
            print('Config found in logs.')

    return train_logs, val_logs


def mmrotate_clear_checkpoints(path_logging_dir, val_logs):

    print('Epoch Number :', len(val_logs['mAP']))
    print('Max mAP :', max(val_logs['mAP']))
    max_index = np.argmax(val_logs['mAP'])
    print('Best Epoch:', max_index)
    print('mAP:', val_logs['mAP'][max_index])

    first_epoch = 1
    best_epoch = max_index + 1
    last_epoch = len(val_logs['mAP'])
    
    if best_epoch == last_epoch:
        keep_epochs = {
            first_epoch: 'model_first.pth',
            best_epoch: 'model_best.pth',
        }
    else:
        keep_epochs = {
            first_epoch: 'model_first.pth',
            best_epoch: 'model_best.pth',
            last_epoch: 'model_last.pth',
        }

    for filename in os.listdir(path_logging_dir):
        if not filename.startswith('epoch_') or not filename.endswith('.pth'):
            continue  # Skip files like latest.pth or model_best.pth
        parts = filename.split('_')
        try:
            epoch_num = int(parts[1].split('.')[0])
        except (IndexError, ValueError):
            print(f"Skipping file (can't parse epoch): {filename}")
            continue

        file_path = os.path.join(path_logging_dir, filename)
        if epoch_num in keep_epochs:
            new_name = keep_epochs[epoch_num]
            new_path = os.path.join(path_logging_dir, new_name)
            os.rename(file_path, new_path)
            print(f"Renamed: {filename} → {new_name}")
        else:
            os.remove(file_path)
            print(f"Deleted: {filename}")


# ---------------------- #
#                        #
#      mmyolo utils      #
#                        #
# ---------------------- #

def mmyolo_get_log_data_path(path_logging_dir):

    latest_folder = max(
        [os.path.join(path_logging_dir, d) for d in os.listdir(path_logging_dir) if os.path.isdir(os.path.join(path_logging_dir, d))],
        key=os.path.getmtime
    )

    # Now, construct the path to the 'vis_data' Scala JSON file
    vis_data_json_path = os.path.join(latest_folder, "vis_data", "scalars.json")
    return vis_data_json_path

def mmyolo_get_logs(vis_data_json_path):
    # Read and load logs from the JSON file
    with open(os.path.join(vis_data_json_path), 'r') as file:
        logs = [json.loads(line) for line in file]
        
        # Separate logs based on specific keys
        train_logs = [log for log in logs if "base_lr" in log]
        val_logs = [log for log in logs if "coco/bbox_mAP" in log]
        
        # Organize training logs into a dictionary of lists
        train_logs = {key: [log[key] for log in train_logs] for key in train_logs[0]}
        print('train_logs', train_logs.keys())
        print('train_logs length:', len(train_logs['epoch']))

        try:
            # Organize validation logs into a dictionary of lists
            val_logs = {key: [log[key] for log in val_logs] for key in val_logs[0]}
            print('val_logs', val_logs.keys())
        except IndexError:
            print('no val for now')

    return train_logs, val_logs


# ---------------------- #
#                        #
#   detectron2 utils     #
#                        #
# ---------------------- #

def detectron2_get_log_data_path(path_logging_dir):
    return os.path.join(path_logging_dir, "metrics.json")


def detectron2_get_logs(vis_data_json_path):
    # Read and load logs from the JSON file
    with open(os.path.join(vis_data_json_path), 'r') as file:
        logs = [json.loads(line) for line in file]

    # Séparer les logs d'entraînement et de validation
    train_logs = [log for log in logs if "iteration" in log]
    val_logs = [log for log in logs if "coco/bbox_mAP" in log]

    # Récupérer toutes les clés possibles
    all_keys = set(key for log in train_logs for key in log.keys())

    # Organiser les logs d'entraînement en dictionnaire
    train_logs_dict = {key: [log.get(key, None) for log in train_logs] for key in all_keys}

    print('train_logs', train_logs_dict.keys())
    print('train_logs length:', len(train_logs_dict["iteration"]))

    try:
        all_val_keys = set(key for log in val_logs for key in log.keys())
        val_logs_dict = {key: [log.get(key, None) for log in val_logs] for key in all_val_keys}
        print('val_logs', val_logs_dict.keys())
    except IndexError:
        val_logs_dict = {}
        print('no val for now')

    return train_logs_dict, val_logs_dict

def detectron2_plot_metrics_vs_steps(formatted_logs, keys):
    
    if isinstance(keys, str):
        keys = [keys]
        
    steps = formatted_logs["iteration"]
    # Plot each metric
    plt.figure(figsize=(15, 8))
    for key in keys:
        if key in formatted_logs:
            metric_values = formatted_logs[key]
            plt.plot(steps, metric_values, marker='o', linestyle='-', label=f'{key.capitalize()}')

    # Add labels, title, legend, and grid
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Metrics as a Function of Steps", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Display the plot
    plt.show()


# ---------------------- #
#                        #
#      VISU utils        #
#                        #
# ---------------------- #
import matplotlib.cm as cm

def check(
        id, 
        save_dir, 
        image_dir,
        show_gt = True,
        show_pred = True,
    ):

    path_ground_truth = os.path.join(save_dir, 'ground_truth.json')
    path_prediction = os.path.join(save_dir, 'predictions.json')

    with open(path_ground_truth, 'r') as f:
        gt_data = json.load(f)

    with open(path_prediction, 'r') as f:
        pred_data = json.load(f)

    # Select the first image_id
    first_image_id = id #gt_data['annotations'][id]['image_id']

    # Extract ground truth segmentations and their category IDs
    gt_segmentations = [(ann['segmentation'], ann['category_id']) for ann in gt_data['annotations'] if ann['image_id'] == first_image_id]
    pred_segmentations = [(res['segmentation'], res['category_id']) for res in pred_data if res['image_id'] == first_image_id]

    # Set a color map to differentiate categories
    cmap = cm.get_cmap('tab10')
    #min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    min_x, min_y, max_x, max_y = 1000, 1000, 0, 0

    # Create the plot
    plt.figure(figsize=(6, 6))

    if image_dir:
        image_name = next(img['file_name'] for img in gt_data['images'] if img['id'] == first_image_id)
        img_path = os.path.join(image_dir, image_name)
        img = plt.imread(img_path)
        plt.imshow(img)

    # Plot ground truth segmentations
    if show_gt:
        for segmentation, category_id in gt_segmentations:
            for poly in segmentation:
                x = poly[0::2]  # x-coordinates
                y = poly[1::2]  # y-coordinates
                min_x, min_y = min(min(x), min_x), min(min(y), min_y)
                max_x, max_y = max(max(x), max_x), max(max(y), max_y)
                #plt.fill(x, y, alpha=0.2, edgecolor='black', color=cmap(category_id % 10), linewidth=1, label=f'GT: Category {category_id}')
                plt.plot(x+ [x[0]], y + [y[0]], linestyle='-', color=cmap(category_id % 10), linewidth=1, label=f'Pred: Category {category_id}')

    # Plot predicted segmentations with dotted lines
    if show_pred:
        for segmentation, category_id in pred_segmentations:
            for poly in segmentation:
                x = poly[0::2]  # x-coordinates
                y = poly[1::2]  # y-coordinates
                min_x, min_y = min(min(x), min_x), min(min(y), min_y)
                max_x, max_y = max(max(x), max_x), max(max(y), max_y)
                plt.plot(x + [x[0]], y + [y[0]], linestyle='--', color=cmap(category_id % 10), linewidth=1, label=f'Pred: Category {category_id}')

    # Customize the plot
    #plt.title(f'Segmentations for Image ID: {first_image_id}')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Invert Y-axis for correct orientation
    #plt.legend(loc='upper right')
    plt.grid(False)  # Disable grid
    plt.axis('equal')
    plt.axis('off')
    plt.axis([min_x - 10, max_x + 10, max_y + 10, min_y - 10])
    plt.tight_layout()
    plt.show()