from faster_live_portrait import FasterLivePortraitPipeline
from omegaconf import OmegaConf
from .config import get_live_portrait_config
import numpy as np
import torch
import cv2

class FasterLivePortrait:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "target": ("IMAGE",),
            },
            "optional": {
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "FasterLivePortrait"
    def __init__(self):
        config_dict = get_live_portrait_config()
        self.pipeline = FasterLivePortraitPipeline(cfg=OmegaConf.create(config_dict), is_animal=False)

    def process_image(self, source, target):
        source_np = tensor_to_cv2(source)
        target_np = tensor_to_cv2(target)
        processed_image = self.pipeline.animate_image(source_np, target_np)
        if processed_image is None:
            tensor = torch.from_numpy(source_np.astype(np.float32) / 255.0)
        else:
            tensor = torch.from_numpy(processed_image.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0)
        return (tensor,)
    
def tensor_to_cv2(tensor):
    arr = tensor.detach().cpu().numpy()
    # Remove batch dimension if present (N, C, H, W)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]  # now (C, H, W)

    # If shape is (C, H, W), convert to (H, W, C)
    # if arr.ndim == 3 and arr.shape[0] in [1, 3]:
    #     arr = np.transpose(arr, (1, 2, 0))

    arr = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # Convert from float [0,1] to uint8 [0,255] if needed
    if arr.dtype in [np.float32, np.float64]:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr

NODE_CLASS_MAPPINGS = {
    "FasterLivePortrait": FasterLivePortrait,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FasterLivePortrait": "FasterLivePortrait",
}