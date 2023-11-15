# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import cv2
from PIL import Image
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.image_face_fusion = pipeline(
            Tasks.image_face_fusion, 
            model='damo/cv_unet-image-face-fusion_damo'
        )

    def predict(
        self,
        template_image: Path = Input(description="Input body image"),
        user_image: Path = Input(description="Input face image"),
    ) -> Path:
        """Run a single prediction on the model"""
        pil_template = Image.open(template_image)
        pil_user = Image.open(user_image)
        result = self.image_face_fusion(dict(template=pil_template, user=pil_user))
        output_path = "output.png"
        cv2.imwrite(output_path, result[OutputKeys.OUTPUT_IMG])
        return Path(output_path)
