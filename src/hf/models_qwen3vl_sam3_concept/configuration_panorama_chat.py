"""HF config for PANORAMA: the Qwen3-VL config plus the chat template and the selection
threshold."""
from transformers.utils import logging

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

logger = logging.get_logger(__name__)


class PanoramaChatConfigQwen(Qwen3VLConfig):
    model_type = 'panorama_chat'

    def __init__(
            self,
            template=None,
            score_threshold=0.5,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.template = template
        # Keep proposals with sigmoid(match score) > score_threshold; top-1 if none.
        self.score_threshold = score_threshold

    def to_dict(self):
        """Adds the PANORAMA fields."""
        output = super().to_dict()
        output["template"] = self.template
        output["score_threshold"] = self.score_threshold
        return output
