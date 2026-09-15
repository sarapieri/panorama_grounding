"""SAM 3 concept detector for PANORAMA: the projected [SEG] replaces SAM 3's text features, giving
per-phrase mask proposals, query embeddings and semantic logits."""
import os
from types import SimpleNamespace

import torch
from mmengine.model import BaseModule

from third_parts.sam3_concept.build_image import build_sam3_image_concept

# Default folder for a relative ckpt_path; set PANORAMA_SAM3_PATH in .env.
BASE_DIR = 'pretrained/sam3'


def _load_concept_checkpoint(model, ckpt_path):
    """Load the detector.* weights of sam3.pt, skipping the text tower."""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    for k in ('model', 'model_state_dict', 'state_dict'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break

    det = {}
    for k, v in sd.items():
        kk = k[len('module.'):] if k.startswith('module.') else k
        if not kk.startswith('detector.'):
            continue
        kk = kk[len('detector.'):]
        if kk.startswith('backbone.language_backbone.'):
            continue
        det[kk] = v

    missing, unexpected = model.load_state_dict(det, strict=False)
    print(f"[SAM3-C] loaded {len(det)} detector keys from {ckpt_path}; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"[SAM3-C] WARNING missing={list(missing)[:8]}{'...' if len(missing) > 8 else ''} "
              f"unexpected={list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''} -- "
              "verify the checkpoint matches the concept build.")


class Sam3ConceptTrainRunner(BaseModule):
    def __init__(self, ckpt_path: str = "sam3.pt", load_checkpoint: bool = True, device=None):
        super().__init__(init_cfg=None)

        full_path = (ckpt_path if os.path.isabs(ckpt_path)
                     else os.path.join(BASE_DIR, ckpt_path))

        # The HF model builds on CPU (meta-safe under from_pretrained); training uses the default device.
        build_kwargs = {} if device is None else {"device": device}
        self.image_model = build_sam3_image_concept(**build_kwargs)

        if load_checkpoint:
            with torch.no_grad():
                _load_concept_checkpoint(self.image_model, full_path)

        self.hidden_dim = 256
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image / 255.
        img_mean = torch.tensor(self.img_mean, dtype=image.dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=image.dtype, device=image.device)[:, None, None]
        image = (image - img_mean) / img_std
        return image

    def encode_image(self, images: torch.Tensor):
        """Frozen ViT backbone, run once per batch of images."""
        m = self.image_model
        backbone_out = {"img_batch_all_stages": images}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            backbone_out.update(m.backbone.forward_image(images))
        return backbone_out

    def forward_prompts(self, backbone_out, concept_tokens: torch.Tensor, img_ids=None):
        """concept_tokens (P, 256): one projected [SEG] per prompt; img_ids (P,): image index of each
        prompt (None when P == B). Returns pred_masks (P, Q, h, w), pred_logits (P, Q, 1), pred_boxes
        (P, Q, 4), queries (P, Q, 256) and semantic_seg (P, 1, h, w)."""
        m = self.image_model
        backbone_out = dict(backbone_out)
        P = concept_tokens.shape[0]
        device = concept_tokens.device
        if img_ids is None:
            B = backbone_out["img_batch_all_stages"].shape[0]
            assert P == B, "img_ids required when #concepts != #images"
            img_ids = torch.arange(B, device=device)

        # The [SEG] replaces the text features: one token per prompt.
        concept = concept_tokens.unsqueeze(0).to(backbone_out["vision_features"].dtype)
        backbone_out["language_features"] = concept
        backbone_out["language_mask"] = torch.zeros(P, 1, dtype=torch.bool, device=device)

        find_input = SimpleNamespace(
            text_ids=torch.arange(P, device=device),
            img_ids=img_ids,
            input_points=None,
        )
        geometric_prompt = m._get_dummy_prompt(num_prompts=P)

        # Fusion, decoder and heads run with grad.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prompt, prompt_mask, backbone_out = m._encode_prompt(
                backbone_out, find_input, geometric_prompt,
            )
            backbone_out, encoder_out, _ = m._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {"encoder_out": encoder_out, "backbone_out": backbone_out},
            }
            out, hs = m._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
            m._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )
        return out

    def forward_concept(self, images: torch.Tensor, concept_tokens: torch.Tensor, img_ids=None):
        """encode_image then forward_prompts."""
        return self.forward_prompts(self.encode_image(images), concept_tokens, img_ids=img_ids)

    def forward(self, batch):
        raise NotImplementedError
