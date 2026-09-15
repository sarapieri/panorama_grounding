# Shared dataset plumbing: Qwen3-VL image processing, grounding-encoder input, conversation
# formatting and tokenization, fractional repeats and refetch-on-failure indexing.
# Adapted from Sa2VA (Yuan et al., 2025, https://github.com/bytedance/Sa2VA).
from functools import partial
from typing import Literal, Optional, Dict, List, Any
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from mmengine import print_log
from xtuner.registry import BUILDER
from .data_utils import template_map_fn, tokenize_conversation


class PanoramaDatasetMixin:
    """
    Mixin with the functionality shared by the PANORAMA datasets: image processing through the
    Qwen3-VL processor, the grounding-encoder image, conversation formatting and tokenization.
    """

    def _init_architecture_config(self, arch_type: Literal['qwen'] = 'qwen'):
        """Image placeholder tokens of the VLM (only Qwen3-VL is supported)."""
        assert arch_type == 'qwen', f"only arch_type='qwen' is supported, got {arch_type!r}"
        self.arch_type = arch_type
        self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
        self.IMG_START_TOKEN = '<|vision_start|>'
        self.IMG_END_TOKEN = '<|vision_end|>'

    def _init_image_processing_config(self):
        """Pixel budget handed to the Qwen3-VL image processor."""
        self.min_pixels_single = 512 * 28 * 28
        self.max_pixels_single = 2048 * 28 * 28

    def _init_tokenizer(self, tokenizer_config, special_tokens: Optional[List[str]] = None):
        """Initialize tokenizer with special tokens."""
        self.tokenizer = BUILDER.build(tokenizer_config)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

    def _init_image_processor(self, preprocessor_config):
        """The Qwen3-VL processor (tokenizer + image processor) that produces pixel_values."""
        assert preprocessor_config is not None, "a Qwen3-VL preprocessor config is required"
        self.preprocessor = BUILDER.build(preprocessor_config)

    def _init_extra_image_processor(self, extra_image_processor_config=None):
        """Initialize extra image processor for grounding."""
        if extra_image_processor_config is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor_config)
        else:
            self.extra_image_processor = None

    def _process_single_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process one image: the grounding-encoder input (g_pixel_values) and the VLM
        pixel_values / image_grid_thw, plus the number of image tokens.
        """
        result = {}

        # Process for grounding if needed
        if getattr(self, 'extra_image_processor', None) is not None:
            g_image = np.array(image)
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            result['g_pixel_values'] = g_pixel_values

        merge_length = self.preprocessor.image_processor.merge_size ** 2
        _data_dict = self.preprocessor.image_processor(
            images=[image], min_pixels=self.min_pixels_single, max_pixels=self.max_pixels_single
        )
        num_image_tokens = int(_data_dict['image_grid_thw'][0].prod()) // merge_length
        result.update(_data_dict)

        result['num_image_tokens'] = num_image_tokens
        return result

    def _create_image_token_string(self, num_image_tokens: int) -> str:
        """Image placeholder string for the given number of image tokens."""
        return f'{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * num_image_tokens}{self.IMG_END_TOKEN}'

    def _process_conversations_for_encoding(self, conversations: List[Dict],
                                            image_token_str: Optional[str] = None) -> List[Dict]:
        """
        Turn from/value conversation messages into input/output turns, replacing the <image>
        placeholder of the first human turn with the image token string.
        """
        # Already in the correct format
        if conversations and 'input' in conversations[0] and 'output' in conversations[0]:
            return conversations

        input_text = ''
        out_conversation = []

        # Skip leading GPT messages
        while conversations and conversations[0]['from'] == 'gpt':
            conversations = conversations[1:]

        conv_idx = 0
        for msg in conversations:
            if msg['from'] == 'human':
                value = msg['value']

                # Handle image token replacement
                if '<image>' in value:
                    if image_token_str is None:
                        value = value.replace('<image>', '')
                    else:
                        assert conv_idx == 0, f"Expected conversation index to be 0, but got {conv_idx} / {value}"
                        value = value.replace('<image>', image_token_str)
                        value = value.strip()

                input_text += value
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input_text,
                    'output': msg['value'].strip()
                })
                input_text = ''
            else:
                raise NotImplementedError(f"Unknown message role: {msg['from']}")

            conv_idx += 1

        return out_conversation

    def get_inputid_labels(self, conversations: List[Dict]) -> Dict[str, List]:
        """
        Convert conversations to input_ids and labels for training (template_map_fn, then
        tokenization with the chat template).
        """
        data_dict = {'conversation': conversations}
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = tokenize_conversation(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        return result

    def _get_modality_length_default(self, length: int = 100) -> int:
        """Get default modality length."""
        return length

    def _read_image(self, image_path: str) -> Optional[Image.Image]:
        """Read an image as RGB; None (logged) if it cannot be read."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print_log(f'Error reading image {image_path}: {e}', logger='current')
            return None


class PanoramaBaseDataset(Dataset, PanoramaDatasetMixin):
    """
    Base dataset class for PANORAMA datasets: common initialization, fractional repeats and
    refetch-on-failure indexing.
    """

    def __init__(self,
                 tokenizer,
                 prompt_template,
                 max_length: int = 2048,
                 special_tokens: Optional[List[str]] = None,
                 arch_type: Literal['qwen'] = 'qwen',
                 preprocessor=None,
                 extra_image_processor=None,
                 max_refetch: int = 1000,
                 repeats: float = 1.0,
                 name: str = "PanoramaBaseDataset",
                 ):
        """
        Args:
            tokenizer: Tokenizer configuration
            prompt_template: Template for formatting prompts
            max_length: Maximum sequence length
            special_tokens: List of special tokens to add
            arch_type: VLM family (only 'qwen')
            preprocessor: Qwen3-VL processor configuration
            extra_image_processor: Extra image processor for grounding
            max_refetch: Maximum refetch attempts
            repeats: Number of times to repeat the dataset (can be fractional, e.g., 0.2)
        """
        super().__init__()

        # Store core configurations
        self.template = prompt_template
        self.max_length = max_length
        self._max_refetch = max_refetch
        self.repeats = repeats

        # Pre-compute index mapping for equal distribution when using fractional repeats
        self._index_mapping = None

        # Template mapping function for format conversion
        self.template_map_fn = partial(template_map_fn, template=self.template)

        # Set name, it is for logging purposes
        self.name = name

        # Initialize architecture and processing configs
        self._init_architecture_config(arch_type)
        self._init_image_processing_config()

        # Initialize processors
        self._init_tokenizer(tokenizer, special_tokens)
        self._init_image_processor(preprocessor)
        self._init_extra_image_processor(extra_image_processor)

    def __len__(self):
        """Get total length considering repeats."""
        return int(self.real_len() * self.repeats)

    def real_len(self):
        """Get the actual length without repeats. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement real_len")

    def _get_index_mapping(self):
        """Create or return cached index mapping for shuffled samples with fractional repeats."""
        if self._index_mapping is None:
            real_length = self.real_len()
            total_length = int(real_length * self.repeats)

            # Create indices based on repeats
            if self.repeats >= 1.0:
                # For repeats >= 1, repeat indices and take the first total_length
                repeated_indice = np.tile(np.arange(real_length), int(np.ceil(self.repeats)))
                indices = np.random.permutation(repeated_indice)[:total_length]
            else:
                # For repeats < 1, randomly sample total_length indices from all available indices
                indices = np.random.choice(real_length, size=total_length, replace=False)

            self._index_mapping = indices

        return self._index_mapping

    def __getitem__(self, index):
        """Unified __getitem__ implementation with refetch logic."""
        # Handle repeats using index mapping for equal distribution
        index_mapping = self._get_index_mapping()
        mapped_index = index_mapping[index]

        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(mapped_index)
            # Broken images may cause the returned data to be None
            if data is None:
                mapped_index = self._rand_another_index()
                continue
            return data

        # If we reach here, all retries failed
        raise RuntimeError(f"Failed to get valid data after {self._max_refetch + 1} attempts")

    def _rand_another_index(self) -> int:
        """Get random index for refetching."""
        return np.random.randint(0, self.real_len())

    def prepare_data(self, index):
        """Prepare data for a given index. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement prepare_data")

    @property
    def modality_length(self):
        """Get modality length for all items."""
        return [self._get_modality_length_default() for _ in range(len(self))]
