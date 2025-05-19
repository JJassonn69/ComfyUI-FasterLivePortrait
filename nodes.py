import os
import cv2
import torch
import logging
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from omegaconf import OmegaConf
from .config import get_live_portrait_config


def load_image(image_source):
    if image_source.startswith('http'):
        logging.info(f"Downloading image from URL: {image_source}")
        response = requests.get(image_source, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    else:
        logging.info(f"Loading image from local path: {image_source}")
        img = Image.open(image_source)
    numpy_rgb = np.array(img)
    source_np = cv2.cvtColor(numpy_rgb, cv2.COLOR_RGB2BGR)
    source_np = cv2.resize(source_np, (512, 512), interpolation=cv2.INTER_LINEAR)
    return source_np

class FasterLivePortraitFastLip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "target": ("IMAGE",),
            },
            "optional": {
                "flag_normalize_lip": ("BOOLEAN", {"default": False}),
                "flag_source_video_eye_retargeting": ("BOOLEAN", {"default": False}),
                "flag_video_editing_head_rotation": ("BOOLEAN", {"default": False}),
                "flag_eye_retargeting": ("BOOLEAN", {"default": False}),
                "flag_lip_retargeting": ("BOOLEAN", {"default": False}),
                "flag_stitching": ("BOOLEAN", {"default": True}),
                "flag_pasteback": ("BOOLEAN", {"default": True}),
                "flag_do_crop": ("BOOLEAN", {"default": True}),
                "flag_do_rot": ("BOOLEAN", {"default": True}),
                "lip_normalize_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "driving_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "animation_region": (["lip", "all"], {"default": "lip"}), # currently only works for lip
                "cfg_mode": (["incremental", "reference"], {"default": "incremental"}),
                "cfg_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "source_max_dim": ("INT", {"default": 1280, "min": 64, "step": 8}),
                "source_division": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1})
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "FasterLivePortrait"
    def __init__(self):
        from faster_live_portrait import FasterLivePortraitPipeline

        config_dict = get_live_portrait_config()
        self.pipeline = FasterLivePortraitPipeline(cfg=OmegaConf.create(config_dict), is_animal=False)

    def process_image(self, source, target, **kwargs):
        self.pipeline.update_cfg(kwargs)
        source_np = tensor_to_cv2(source)
        target_np = tensor_to_cv2(target)
        processed_image = self.pipeline.animate_image(source_np, target_np)
        if processed_image is None:
            tensor = torch.from_numpy(source_np.astype(np.float32) / 255.0)
        else:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
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

class FasterLivePortraitStandard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_url": ("STRING", {"default": ""}),
                "target": ("IMAGE",),
            },
            "optional": {
                "flag_normalize_lip": ("BOOLEAN", {"default": False}),
                "flag_source_video_eye_retargeting": ("BOOLEAN", {"default": True}),
                "flag_video_editing_head_rotation": ("BOOLEAN", {"default": False}),
                "flag_eye_retargeting": ("BOOLEAN", {"default": False}),
                "flag_lip_retargeting": ("BOOLEAN", {"default": True}),
                "flag_stitching": ("BOOLEAN", {"default": True}),
                "flag_pasteback": ("BOOLEAN", {"default": True}),
                "flag_do_crop": ("BOOLEAN", {"default": True}),
                "flag_do_rot": ("BOOLEAN", {"default": True}),
                "lip_normalize_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "driving_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "animation_region": (["exp", "pose", "lip", "eyes", "all"], {"default": "all"}),
                "cfg_mode": (["incremental", "reference"], {"default": "incremental"}),
                "cfg_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "source_max_dim": ("INT", {"default": 1280, "min": 64, "step": 8}),
                "source_division": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1})
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "FasterLivePortrait"
    def __init__(self):
        from faster_live_portrait_std import FasterLivePortraitPipeline
        
        config_dict = get_live_portrait_config()
        self.pipeline = FasterLivePortraitPipeline(cfg=OmegaConf.create(config_dict), is_animal=False)
        self.first_frame = True
        self.previous_source_url = None

    def process_image(self, source_url, target, **kwargs):
        self.pipeline.update_cfg(kwargs)

        if not source_url or not (source_url.startswith('http://') or source_url.startswith('https://')):
            logging.error(f"Invalid or empty source URL: '{source_url}'. Returning target image.")
            return (target,)

        if self.previous_source_url != source_url:
            try:
                logging.info(f"Source URL changed or not yet prepared. Loading from: {source_url}")
                source_np = load_image(source_url)
            except Exception as e:
                logging.error(f"Error loading or processing source image from URL '{source_url}': {e}")
                return (target,)

            ret = self.pipeline.prepare_source(img_bgr=source_np)
            if ret is None:
                logging.error(f"Failed to prepare source: No face detected in source image from '{source_url}' or other preparation error.")
                return (target,)
            
            self.previous_source_url = source_url
            self.first_frame = True

        target_cv2_rgb_resized = tensor_to_cv2(target)
        target_np_for_pipeline = cv2.cvtColor(target_cv2_rgb_resized, cv2.COLOR_RGB2BGR)
             
        _, _, processed_image_pipeline, _ = self.pipeline.run(
            target_np_for_pipeline,
            self.pipeline.src_imgs[0], 
            self.pipeline.src_infos[0], 
            first_frame=self.first_frame
        )
        self.first_frame = False

        output_tensor = torch.from_numpy(processed_image_pipeline.astype(np.float32) / 255.0)
        output_tensor = output_tensor.unsqueeze(0)
        return (output_tensor,)
    
NODE_CLASS_MAPPINGS = {
    "FasterLivePortraitFastLip": FasterLivePortraitFastLip,
    "FasterLivePortraitStandard": FasterLivePortraitStandard,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FasterLivePortraitFastLip": "FasterLivePortraitFastLip",
    "FasterLivePortraitStandard": "FasterLivePortraitStandard",
}