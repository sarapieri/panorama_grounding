"""Shared PANORAMA plumbing: VLM and tokenizer build, [SEG] token, LoRA, [SEG] bridge, checkpoint
I/O and parameter logs."""
from typing import Literal
from collections import OrderedDict
import os.path as osp
import torch
import torch.nn as nn

from mmengine.model import BaseModel
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from third_parts.mmdet.models.utils.point_sample import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness

from peft import PeftModelForCausalLM

from transformers import AutoImageProcessor, AutoVideoProcessor


def _load_pretrained_state_dict(pth_model):
    """Load a .pth with weights_only=False (mmengine checkpoints hold non-tensor state that
    torch>=2.6 refuses by default); directories go through xtuner."""
    if osp.isfile(pth_model):
        state_dict = torch.load(pth_model, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        return state_dict
    return guess_load_checkpoint(pth_model)


class PanoramaBase(BaseModel):

    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 template=None,
                 arch_type:Literal['qwen']='qwen',
                 training_bs:int=0,
                 # SAM 3 input size.
                 grounding_img_size:int=1008,
                 # Max phrases supervised per image.
                 max_objs_per_image:int=10,
                 # False freezes the lm_head/embed_tokens copies (LoRA-only finetune); keep
                 # modules_to_save in the config so their checkpoint keys still load.
                 train_token_embeddings:bool=True,
                 ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['[SEG]']

        self.max_objs_per_image = max_objs_per_image

        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens(tokenizer, special_tokens)

        if arch_type == 'qwen':
            image_processor = AutoImageProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            video_processor = AutoVideoProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            self.mllm._init_processor(image_processor, video_processor)

        # The proposal model starts fully frozen; Panorama unfreezes the parts it trains.
        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)

        # Untie embed_tokens and lm_head so the new token rows train separately.
        if self.arch_type == 'qwen' and self.mllm.model.config.tie_word_embeddings:
            print("Untying embed_tokens and lm_head weights for Qwen model.")
            self.mllm.model.config.tie_word_embeddings = False
            lm_head = self.mllm.model.get_output_embeddings()
            if lm_head is not None:
                input_embeddings = self.mllm.model.get_input_embeddings()
                lm_head.weight = nn.Parameter(input_embeddings.weight.clone())

        in_dim = self.mllm.get_embedding_size()
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        # Zero-init the final projection: proposals start from SAM 3's unconditioned prior, which
        # keeps the bridge robust to the seed (a bad init can saturate the mask path early).
        nn.init.zeros_(self.text_hidden_fcs[2].weight)
        nn.init.zeros_(self.text_hidden_fcs[2].bias)
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.torch_dtype = torch_dtype

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.template = template
        self.bs = training_bs
        self.grounding_img_size = grounding_img_size

        if self.mllm.use_llm_lora:
            self.mllm.manual_prepare_llm_for_lora()

        # Load after the LoRA setup, otherwise the adapter and modules_to_save keys are dropped.
        if pretrained_pth is not None:
            pretrained_state_dict = _load_pretrained_state_dict(pretrained_pth)
            _incompat = self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')
            # strict=False hides a key mismatch, so report the counts and stop if almost nothing loaded.
            _ckpt_keys = set(pretrained_state_dict)
            _loaded = len(_ckpt_keys) - len(getattr(_incompat, 'unexpected_keys', []) or [])
            print(f'  matched {_loaded}/{len(_ckpt_keys)} checkpoint tensors; '
                  f'{len(getattr(_incompat, "missing_keys", []) or [])} model params not in the '
                  f'checkpoint (expected: frozen detector + anything newly built)')
            for _k in (getattr(_incompat, 'unexpected_keys', []) or [])[:10]:
                print(f'  UNEXPECTED (in checkpoint, not in model): {_k}')
            assert _loaded > 0.5 * len(_ckpt_keys), (
                f'only {_loaded}/{len(_ckpt_keys)} checkpoint tensors matched this model -- the '
                f'checkpoint and the config disagree (wrong parent run, or a changed architecture). '
                f'Refusing to train from what would be a near-random init.')

            if self.arch_type == 'qwen':
                print("Force updating lm_head weight from pretrained state_dict.")
                lm_head_key = 'mllm.model.lm_head.weight'
                if lm_head_key in pretrained_state_dict:
                    lm_head_weight = pretrained_state_dict[lm_head_key]
                    self.mllm.model.get_output_embeddings().weight.data.copy_(lm_head_weight)
                    print(f"Successfully updated lm_head weight from key: {lm_head_key}")
                else:
                    print(f"Warning: lm_head weight key '{lm_head_key}' not found in pretrained_state_dict.")

        # LoRA-only finetune: freeze the lm_head/embed_tokens copies after the load above.
        if not train_token_embeddings:
            _frozen = sum(p.numel() for n, p in self.mllm.named_parameters()
                          if 'modules_to_save' in n and p.requires_grad)
            for n, p in self.mllm.named_parameters():
                if 'modules_to_save' in n:
                    p.requires_grad_(False)
            print(f'[LoRA-only] froze modules_to_save (lm_head/embed_tokens): '
                  f'{_frozen/1e6:.2f}M params -> frozen; LoRA adapters remain trainable')

        print("\n" + "="*80)
        print("MLLM PARAMETERS")
        print("="*80)

        try:
            base_model = self.mllm.model
            total_params = 0
            trainable_params = 0

            for name, param in base_model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                # Per-parameter dump (one line per VLM parameter):
                # print(f"{name:<60} | {'trainable' if param.requires_grad else 'frozen'} | "
                #       f"Shape: {tuple(param.shape)} | Params: {param.numel():,}")

            print("-" * 80)
            print(f"MLLM-ONLY SUMMARY (excludes grounding_encoder + text_hidden_fcs):")
            print(f"  MLLM total parameters: {total_params:,}")
            print(f"  MLLM trainable parameters: {trainable_params:,}")
            print(f"  MLLM frozen parameters: {total_params - trainable_params:,}")
            print(f"  MLLM trainable ratio: {trainable_params/total_params*100:.2f}%")
            print("=" * 80)

        except Exception as e:
            print(f"Failed to access self.mllm.model: {e}")
            print("Available attributes in self.mllm.model:")
            print([attr for attr in dir(self.mllm.model) if not attr.startswith('_')])

    def _log_whole_model_summary(self):
        """Whole-model parameter totals per top-level module; call after any freeze/unfreeze."""
        print("\n" + "=" * 80)
        print("WHOLE-MODEL PARAMETERS (mllm + grounding encoder)")
        print("=" * 80)
        whole_total = sum(p.numel() for p in self.parameters())
        whole_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        covered = 0
        for mod_name in ("mllm", "grounding_encoder", "text_hidden_fcs"):
            mod = getattr(self, mod_name, None)
            if mod is None:
                continue
            mt = sum(p.numel() for p in mod.parameters())
            mtr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            covered += mt
            print(f"  {mod_name:<18} total {mt:,} | trainable {mtr:,}")
        other = whole_total - covered
        if other:
            print(f"  {'other':<18} total {other:,}")
        print("-" * 80)
        print(f"  WHOLE-MODEL total {whole_total:,} | trainable {whole_train:,} "
              f"| ratio {whole_train/whole_total*100:.2f}%")
        print("=" * 80)

    def _add_special_tokens(self, tokenizer, special_tokens):
        self.mllm.add_special_tokens(tokenizer, special_tokens)
        self.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.tokenizer = tokenizer

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def _merge_lora(self):
        if isinstance(self.mllm.model, PeftModelForCausalLM):
            self.mllm.model = self.mllm.model.merge_and_unload()
            return

        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

    def state_dict(self, *args, **kwargs):
        """Checkpoints hold the VLM and the concept bridge; the frozen proposal model is
        rebuilt from sam3.pt (Panorama adds its trained proposal-model parts on top)."""
        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        state_dict_text = self.text_hidden_fcs.state_dict(*args, prefix=prefix + 'text_hidden_fcs.', **kwargs)
        to_return = OrderedDict()
        to_return.update(state_dict_mllm)
        to_return.update(state_dict_text)
        return to_return

    def _get_pseudo_data(self, dtype, device):
        _sz = self.grounding_img_size
        g_pixel_values = torch.zeros((3, _sz, _sz), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

