
import torch
import cv2

from huggingface_hub import hf_hub_download
from model import BoneAgeNet
from preprocessing import preprocess_image


class BoneAgePredictor:

    def __init__(self, device=None):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Download trained checkpoint from Hugging Face
        model_path = hf_hub_download(
            repo_id="BojjaDivya/boneage-ai-model",
            filename="best_boneage_model.pth"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        self.model = BoneAgeNet().to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.target_mean = float(checkpoint["target_mean"])
        self.target_std = float(checkpoint["target_std"])

    def predict(self, image, sex):

        if sex not in (0, 1):
            raise ValueError("Sex must be 0 or 1")

        image_tensor = (
            preprocess_image(image)
            .unsqueeze(0)
            .to(self.device)
        )

        sex_tensor = torch.tensor(
            [[float(sex)]],
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():

            output = self.model(
                image_tensor,
                sex_tensor
            )

        predicted_age = (
            output.item() * self.target_std
            + self.target_mean
        )

        return max(0.0, predicted_age)


def load_image(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError("Unable to load X-ray image")

    return image
