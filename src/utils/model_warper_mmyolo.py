from mmdet.apis import init_detector, inference_detector
import torch
import numpy as np

class ModelWarper():

    def __init__(self, config_file, checkpoint_file, device='cuda:0'):
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
        
        self.model = init_detector(config_file, checkpoint_file, device=device)


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

        result = inference_detector(self.model, image)
        pred_instances = result.pred_instances
        pred_bboxes = pred_instances.bboxes.cpu().numpy()
        pred_labels = pred_instances.labels.cpu().numpy() + 1
        pred_scores = pred_instances.scores.cpu().numpy()

        return {
            'positions': pred_bboxes,
            'scores': pred_scores,
            'classes': pred_labels,
        }