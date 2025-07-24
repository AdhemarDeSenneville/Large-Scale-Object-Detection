from detectron2.engine.defaults import DefaultPredictor
import numpy as np
from detectron2.config import get_cfg


class FasterRCNNAPIWrapper:
    def __init__(self, config_file, checkpoint_file, device = 'cuda:0', path_logging_dir = None):
        """
        Initializes the Faster R-CNN model with the given configuration and weights.
        
        Args:
            config_path (str): Path to model configuration.
            model_weights (str): Path to model weights.
            device (str, optional): CUDA device.
        """

        print("Initializing the model...")
        print(f"config_file: {config_file}")
        print(f"checkpoint_file: {checkpoint_file}")
        print(f"device: {device}")

        # Load configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.cfg.defrost() 
        self.cfg.DATASETS.TRAIN = ("biodigester_train",)
        self.cfg.DATASETS.TEST = ("biodigester_val",)
        self.cfg.MODEL.WEIGHTS = checkpoint_file
        if path_logging_dir:
            self.cfg.OUTPUT_DIR = path_logging_dir
        self.cfg.MODEL.DEVICE = device
        self.cfg.freeze()
        
        # Initialize the predictor
        self.predictor = DefaultPredictor(self.cfg)
    
    def __call__(self, image):
        """
        Object detection on the input image.
        
        Args:
            image (numpy.ndarray or PIL.Image): Input image for inference.
        
        Returns:
            dict: A dictionary containing detected bounding boxes, confidence scores, 
                  and class labels.
                  {
                      'positions': numpy.ndarray of shape (N, 4), bounding box coordinates,
                      'scores': numpy.ndarray of shape (N,), confidence scores,
                      'classes': numpy.ndarray of shape (N,), predicted class labels.
                  }
        """

        image = np.array(image)
        outputs = self.predictor(image)
        
        instances = outputs["instances"]
        pred_bboxes = instances.pred_boxes.tensor.cpu().numpy()
        pred_scores = instances.scores.cpu().numpy()
        pred_labels = instances.pred_classes.cpu().numpy() + 1
        
        return {
            'positions': pred_bboxes,
            'scores': pred_scores,
            'classes': pred_labels,
        }
    