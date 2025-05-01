import os
import folder_paths

def get_live_portrait_config():
    models_dir = folder_paths.models_dir
    ComfyUI_FasterLivePortrait_dir = os.path.abspath(os.path.join(models_dir, "../custom_nodes/ComfyUI-FasterLivePortrait"))

    config_dict = {
        "grid_sample_plugin_path": os.path.join(models_dir, "liveportrait/libgrid_sample_3d_plugin.so"),
        "models": {
            "warping_spade": {
                "name": "WarpingSpadeModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait/warping_spade-fix.trt")
            },
            "motion_extractor": {
                "name": "MotionExtractorModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait/motion_extractor.trt")
            },
            "landmark": {
                "name": "LandmarkModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait/landmark.trt")
            },
            "face_analysis": {
                "name": "FaceAnalysisModel",
                "predict_type": "trt",
                "model_path": [
                    os.path.join(models_dir, "liveportrait/retinaface_det_static.trt"),
                    os.path.join(models_dir, "liveportrait/face_2dpose_106_static.trt")
                ]
            },
            "app_feat_extractor": {
                "name": "AppearanceFeatureExtractorModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait/appearance_feature_extractor.trt")
            },
            "stitching": {
                "name": "StitchingModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait/stitching.trt")
            },
            "stitching_eye_retarget": {
                "name": "StitchingModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait/stitching_eye.trt")
            },
            "stitching_lip_retarget": {
                "name": "StitchingModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait/stitching_lip.trt")
            }
        },
        "animal_models": {
            "warping_spade": {
                "name": "WarpingSpadeModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait_animal_onnx/warping_spade-fix-v1.1.trt")
            },
            "motion_extractor": {
                "name": "MotionExtractorModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait_animal_onnx/motion_extractor-v1.1.trt")
            },
            "app_feat_extractor": {
                "name": "AppearanceFeatureExtractorModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait_animal_onnx/appearance_feature_extractor-v1.1.trt")
            },
            "stitching": {
                "name": "StitchingModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait_animal_onnx/stitching-v1.1.trt")
            },
            "stitching_eye_retarget": {
                "name": "StitchingModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait_animal_onnx/stitching_eye-v1.1.trt")
            },
            "stitching_lip_retarget": {
                "name": "StitchingModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait_animal_onnx/stitching_lip-v1.1.trt")
            },
            "landmark": {
                "name": "LandmarkModel",
                "predict_type": "trt",
                "model_path": os.path.join(models_dir, "liveportrait/landmark.trt")
            },
            "face_analysis": {
                "name": "FaceAnalysisModel",
                "predict_type": "trt",
                "model_path": [
                    os.path.join(models_dir, "liveportrait/retinaface_det_static.trt"),
                    os.path.join(models_dir, "liveportrait/face_2dpose_106_static.trt")
                ]
            }
        },
        "joyvasa_models": {
            "motion_model_path": os.path.join(models_dir, "liveportrait/joyvasa_models/motion_generator/motion_generator_hubert_chinese.pt"),
            "audio_model_path": os.path.join(models_dir, "liveportrait/joyvasa_models/chinese-hubert-base"),
            "motion_template_path": os.path.join(models_dir, "liveportrait/joyvasa_models/motion_template/motion_template.pkl")
        },
        "crop_params": {
            "src_dsize": 512,
            "src_scale": 2.3,
            "src_vx_ratio": 0.0,
            "src_vy_ratio": -0.125,
            "dri_scale": 2.2,
            "dri_vx_ratio": 0.0,
            "dri_vy_ratio": -0.1
        },
        "infer_params": {
            "flag_crop_driving_video": False,
            "flag_normalize_lip": False,
            "flag_source_video_eye_retargeting": False,
            "flag_video_editing_head_rotation": False,
            "flag_eye_retargeting": False,
            "flag_lip_retargeting": False,
            "flag_stitching": True,
            "flag_relative_motion": False,
            "flag_pasteback": True,
            "flag_do_crop": True,
            "flag_do_rot": True,
            "lip_normalize_threshold": 0.1,
            "source_video_eye_retargeting_threshold": 0.18,
            "driving_smooth_observation_variance": 1e-07,
            "anchor_frame": 0,
            "mask_crop_path": os.path.join(ComfyUI_FasterLivePortrait_dir, "assets/mask_template.png"),
            "driving_multiplier": 1.0,
            "animation_region": "lip",
            "cfg_mode": "incremental",
            "cfg_scale": 1.2,
            "source_max_dim": 1280,
            "source_division": 2
        }
    }

    return config_dict
