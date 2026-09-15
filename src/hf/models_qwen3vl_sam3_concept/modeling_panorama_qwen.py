"""HF inference model for PANORAMA: Qwen3-VL writes the grounded caption and each projected [SEG]
conditions and selects SAM 3's proposals. The SAM 3 runner is imported from the repo, so run
with the repo on PYTHONPATH."""
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from transformers.modeling_utils import PreTrainedModel
from qwen_vl_utils import process_vision_info

from .configuration_panorama_chat import PanoramaChatConfigQwen

from src.models.sam3_concept_train import Sam3ConceptTrainRunner


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """(H, W, C) uint8 array resized to target_length x target_length."""
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


class PanoramaChatModelQwen(PreTrainedModel):
    config_class = PanoramaChatConfigQwen
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['Qwen3VisionTransformerPretrainedModel', 'Qwen3VLDecoderLayer',
                         'Sam3ConceptTrainRunner']
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    supports_gradient_checkpointing = True

    def __init__(self, config: PanoramaChatConfigQwen, model=None):
        super().__init__(config)
        self.extra_image_processor = DirectResize(target_length=1008)

        self.min_pixels = 512 * 28 * 28
        self.max_pixels = 2048 * 28 * 28

        self.torch_dtype = torch.bfloat16

        self.score_threshold = float(getattr(config, 'score_threshold', 0.5))
        # Phrases per SAM 3 call, to cap memory; <=0 disables chunking.
        _k = int(os.environ.get('PANORAMA_CONCEPT_EVAL_CHUNK', '8'))
        self.concept_eval_chunk = _k if _k > 0 else 10 ** 9
        print(f"[PANORAMA] score_threshold={self.score_threshold} chunk={self.concept_eval_chunk}")

        if model is not None:
            self.model = model
        else:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration(config)

        llm_hidden_size = config.text_config.hidden_size

        # Weights come from the converted checkpoint; built on CPU so from_pretrained can load it
        # lazily.
        self.grounding_encoder = Sam3ConceptTrainRunner(load_checkpoint=False, device="cpu")
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

    @property
    def lm_head(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
    ):
        assert processor is not None
        assert video is None, "PANORAMA inference is image-only"
        self.processor = processor

        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids('[SEG]')

        text = text.replace('<image>', "")

        if image is None and '<image>' not in past_text:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": past_text + text},
                    ],
                }
            ]
            processed_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            mm_inputs = self.processor(
                text=[processed_text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            mm_inputs = mm_inputs.to(self.device)
            ret_masks = []
        else:
            ori_image_size = image.size

            g_image = np.array(image)
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
            g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(g_pixel_values)
            ]).to(self.torch_dtype).to(self.device)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            processed_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            mm_inputs = self.processor(
                text=[processed_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels
            )
            mm_inputs = mm_inputs.to(self.device)
            ret_masks = []

        # Greedy decoding; PANORAMA_MAX_NEW_TOKENS and PANORAMA_REP_PENALTY set the length limit and
        # repetition penalty.
        generate_output = self.model.generate(
            **mm_inputs,
            max_new_tokens=int(os.environ.get('PANORAMA_MAX_NEW_TOKENS', 2048)),
            do_sample=False,
            repetition_penalty=float(os.environ.get('PANORAMA_REP_PENALTY', 1.0)),
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        generate_output_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(mm_inputs.input_ids, generate_output.sequences)
        ]
        predict = self.processor.batch_decode(generate_output_trimmed, skip_special_tokens=False)[0].strip()

        if image is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks}

        # One projected [SEG] per phrase.
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)
        P = all_seg_hidden_states.shape[0]
        w, h = ori_image_size

        ret_instance_masks = []
        phrase_seg_counts = []

        if P > 0:
            K = self.concept_eval_chunk
            # Backbone once; phrases in chunks of K.
            backbone_out = self.grounding_encoder.encode_image(g_pixel_values)
            dps = self.grounding_encoder.image_model.dot_prod_scoring
            for start in range(0, P, K):
                seg = all_seg_hidden_states[start:start + K]
                pc = seg.shape[0]
                img_ids = torch.zeros(pc, dtype=torch.long, device=g_pixel_values.device)
                out = self.grounding_encoder.forward_prompts(backbone_out, seg, img_ids=img_ids)
                pred_masks = out["pred_masks"]
                # Score the queries with SAM 3's scorer, keyed by the same [SEG], as in training.
                queries = out["queries"]
                # autocast needed: queries are fp32, scorer weights bf16.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    seg_logits = dps(
                        queries.unsqueeze(0),
                        seg.unsqueeze(0).to(queries.dtype),
                        torch.zeros(pc, 1, dtype=torch.bool, device=queries.device),
                    )[0].squeeze(-1)
                prob = seg_logits.float().sigmoid()
                for p in range(pc):
                    sel = (prob[p] > self.score_threshold).nonzero(as_tuple=True)[0]
                    if sel.numel() == 0:
                        sel = prob[p].argmax().reshape(1)  # top-1 fallback
                    qm_soft = pred_masks[p][sel].float()
                    qm_soft = F.interpolate(qm_soft.unsqueeze(0), size=(h, w), mode='bilinear',
                                            align_corners=False).squeeze(0).sigmoid()
                    qm = (qm_soft > 0.5).cpu().numpy()
                    ret_masks.append(qm.any(axis=0)[None])  # phrase mask = union of its instances
                    for inst in qm:
                        ret_instance_masks.append(inst[None])
                    phrase_seg_counts.append(int(qm.shape[0]))
                del out, pred_masks, prob

        return {
            'prediction': predict,
            'prediction_masks': ret_masks,
            'prediction_instance_masks': ret_instance_masks,
            'prediction_phrase_seg_counts': phrase_seg_counts,
        }


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]
