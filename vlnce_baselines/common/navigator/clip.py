import os
from PIL import Image
import torch
import clip
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def direction_id_to_angle(direction_id: int) -> str:
    direction_id = int(direction_id) 

    angle = direction_id * 30
    return f"{angle}.0"


def build_image_path(base_dir: str, episode_id: int, step: int, direction_id: int) -> str:

    direction_id = int(direction_id)
    if direction_id == 0:
        filename = "rgb.jpg"
    else:
        angle_str = direction_id_to_angle(direction_id)
        filename = f"rgb_{angle_str}.jpg"

    return os.path.join(base_dir, str(episode_id), str(step), filename)



def load_image(image_path: str) -> Image.Image:

    image = Image.open(image_path).convert("RGB")
    return image


class CLIPSimilarity:
    def __init__(self, model_name="ViT-B/32", image_base_dir="image_show"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.image_base_dir = image_base_dir

    def encode_image(self, image: Image.Image) -> np.ndarray:

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
        return embedding.cpu().numpy()[0]

    def compute_similarity(
        self,
        episode_id: int,
        matched_step: int,
        matched_direction_id: int,
        current_step: int,
        candidate_direction_ids: set
    ) -> tuple:
        ref_path = build_image_path(self.image_base_dir, episode_id, matched_step, matched_direction_id)
        ref_img = load_image(ref_path)
        ref_feat = self.encode_image(ref_img)
        scores = {}
        for direction_id_str in candidate_direction_ids:
            direction_id = int(direction_id_str)
            cand_path = build_image_path(self.image_base_dir, episode_id, current_step, direction_id)
            cand_img = load_image(cand_path)
            cand_feat = self.encode_image(cand_img)
            sim = cosine_similarity([ref_feat], [cand_feat])[0][0]
            scores[direction_id] = sim

        most_similar_direction = max(scores, key=scores.get)
        return most_similar_direction, scores
