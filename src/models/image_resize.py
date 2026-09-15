import numpy as np
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """(H, W, C) uint8 array resized to target_length x target_length."""
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))
