"""Convert a trained PANORAMA .pth checkpoint to an HF folder (weights, config, trust_remote_code
files); launched by scripts/convert.sh. Adapted from Sa2VA's tools/convert_to_hf.py."""
import argparse
import copy
import os
import os.path as osp
import re

import torch
from mmengine.config import Config
from mmengine.dist import master_only
from xtuner.registry import BUILDER


def parse_args():
    parser = argparse.ArgumentParser(description='Convert a trained PANORAMA .pth to HF format')
    parser.add_argument('config', help='config file path (the one used for training).')
    parser.add_argument('pth_model', help='trained .pth checkpoint (a consolidated state_dict).')
    parser.add_argument('--save-path', type=str, default=None, help='output HF model folder')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16',
                        help='dtype of the saved weights (evaluation loads bfloat16)')
    return parser.parse_args()


@master_only
def master_print(msg):
    print(msg)


def main():
    args = parse_args()

    # build the training model from the config
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)

    state_dict = torch.load(args.pth_model, map_location='cpu', weights_only=False)
    state_dict = state_dict['state_dict']
    load_res = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded PTH model from {args.pth_model}')

    # Every trainable grounding param must be in the checkpoint (it is saved with
    # exclude_frozen_parameters, so frozen SAM 3 weights come from the fresh build). Otherwise
    # strict=False would silently export untrained weights.
    _ckpt_keys = set(state_dict.keys())
    _trained_g = [f"grounding_encoder.{n}" for n, p in model.grounding_encoder.named_parameters()
                  if p.requires_grad]
    _missing_g = [k for k in _trained_g if k not in _ckpt_keys]
    print(f"[convert] checkpoint keys={len(_ckpt_keys)}; load missing={len(load_res.missing_keys)} "
          f"unexpected={len(load_res.unexpected_keys)}; trained grounding params={len(_trained_g)}")
    if _missing_g:
        raise SystemExit(
            f"[convert] FATAL: {len(_missing_g)}/{len(_trained_g)} trained grounding params are "
            f"absent from the checkpoint; refusing to export untrained detector weights. "
            f"e.g. {_missing_g[:5]}.")
    if _trained_g:
        print(f"[convert] OK: all {len(_trained_g)} trained grounding params found in checkpoint.")

    iter_str = os.path.basename(args.pth_model).split('.')[0]

    # merge LoRA and prepare the mllm for HF export
    model._merge_lora()
    model.mllm.model.modules_to_save = None
    model.mllm.model.transfer_to_hf = True

    all_state_dict = model.all_state_dict()

    # HF model definition (Qwen3-VL + SAM 3 concept grounding encoder), loaded with trust_remote_code.
    hf_pkg = 'models_qwen3vl_sam3_concept'
    from src.hf.models_qwen3vl_sam3_concept.configuration_panorama_chat import PanoramaChatConfigQwen
    from src.hf.models_qwen3vl_sam3_concept.modeling_panorama_qwen import PanoramaChatModelQwen

    arch_type = cfg.model.get('arch_type', 'qwen')
    assert 'qwen' in arch_type, f"This converter only supports the qwen3 arch, got '{arch_type}'."

    # HF config seeded from the base Qwen3-VL model
    config = PanoramaChatConfigQwen.from_pretrained(cfg.path)
    config_dict = config.to_dict()
    config_dict['text_config']['vocab_size'] = len(model.mllm.tokenizer)
    config_dict['tie_word_embeddings'] = False

    # strip the system-prompt block from the Qwen jinja template
    template_str = cfg.template
    system_prompt_pattern = re.compile(
        r"{% if loop\.first and message\['role'] != 'system' %}.*?<\|im_end\|>\s*{% endif %}",
        re.DOTALL,
    )
    template_str = system_prompt_pattern.sub('', template_str)
    config_dict['template'] = template_str

    config_dict['auto_map'] = {
        'AutoConfig': 'configuration_panorama_chat.PanoramaChatConfigQwen',
        'AutoModel': 'modeling_panorama_qwen.PanoramaChatModelQwen',
        'AutoModelForCausalLM': 'modeling_panorama_qwen.PanoramaChatModelQwen',
    }

    # Selection threshold used at inference (same value as training's score_threshold).
    config_dict['score_threshold'] = float(cfg.model.get('score_threshold', 0.5))

    # remap training keys -> HF keys (drop the 'mllm.' prefix). The HF model reuses the train-time
    # concept runner verbatim, so grounding_encoder keys need no remapping.
    name_map = {'mllm.': ''}
    all_state_dict_new = {}
    for key in all_state_dict.keys():
        new_key = copy.deepcopy(key)
        for _text in name_map.keys():
            new_key = new_key.replace(_text, name_map[_text])
        all_state_dict_new[new_key] = all_state_dict[key]

    hf_config = PanoramaChatConfigQwen(**config_dict)
    hf_config.text_config.tie_word_embeddings = False

    hf_model = PanoramaChatModelQwen(hf_config, model=model.mllm.model)
    missing_keys, unexpected_keys = hf_model.load_state_dict(all_state_dict_new)

    if args.save_path is None:
        args.save_path = f"./{os.path.dirname(args.pth_model)}_{iter_str}_hf"

    hf_model = hf_model.to(getattr(torch, args.dtype))
    hf_model.save_pretrained(args.save_path)
    # qwen uses the processor (tokenizer + image/video processors)
    model.mllm.processor.save_pretrained(args.save_path)

    master_print("\n--- Weight Loading Report ---")
    if missing_keys:
        master_print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        master_print(f"Warning: Unexpected keys: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        master_print("All keys matched successfully!")

    print(f"Saved the HF model into {args.save_path}")

    # copy the trust_remote_code modeling/config files next to the weights
    os.system(f"cp -pr ./src/hf/{hf_pkg}/* {args.save_path}")


if __name__ == '__main__':
    main()
