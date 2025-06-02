import mmrotate
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
        Performs object detection on the input image.
        
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

        obb = []
        scores = []
        classes = []
        
        # image = torch.tensor(image).transpose(1,2,0).to(self.device)  # not needed
        image = np.array(image)[...,::-1] # Important to flip the image
        result = inference_detector(self.model, image)  # .transpose(1,2,0) # not needed

        for id, arr in enumerate(result):
            for row in arr:
                obb.append(row[:5])
                scores.append(row[5])
                classes.append(id + 1)
            
        return {
            'positions': np.array(obb),
            'scores': np.array(scores),
            'classes': np.array(classes),
        }