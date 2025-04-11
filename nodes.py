
class FasterLivePortrait:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGES",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "FasterLivePortrait"

    def process_image(self, image):
       return (image,)
       

NODE_CLASS_MAPPINGS = {
    "FasterLivePortrait": FasterLivePortrait,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FasterLivePortrait": "FasterLivePortrait",
}