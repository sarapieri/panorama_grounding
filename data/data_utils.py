# Tokenization, chat-template mapping, the training collate function and the concat dataset.
# Adapted from xtuner (https://github.com/InternLM/xtuner) and Sa2VA.
import copy
import numpy as np
import torch
from typing import Dict, List, Sequence
from torch.nn.utils.rnn import pad_sequence
from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils import IGNORE_INDEX, DEFAULT_PAD_TOKEN_INDEX
from xtuner.registry import BUILDER
from mmengine.logging import print_log
from torch.utils.data import ConcatDataset as TorchConcatDataset


def tokenize_conversation(
        example,
        tokenizer,
        max_length,
):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        # Add output
        output_with_loss = single_turn_conversation.get(
            'output_with_loss', True)
        output = single_turn_conversation['output']
        output_encode = tokenizer.encode(output, add_special_tokens=False)
        input_ids += output_encode
        if output_with_loss:
            labels += copy.deepcopy(output_encode)
        else:
            labels += [IGNORE_INDEX] * len(output_encode)
        # Add EOS_TOKEN (with loss)
        if single_turn_conversation.get('need_eos_token', True):
            next_needs_bos_token = True
            input_ids += eos_token_id
            if output_with_loss:
                labels += copy.deepcopy(eos_token_id)
            else:
                labels += [IGNORE_INDEX] * len(eos_token_id)
        else:
            next_needs_bos_token = False
        # Add SEP (without loss)
        sep = single_turn_conversation.get('sep', '')
        if sep != '':
            sep_encode = tokenizer.encode(sep, add_special_tokens=False)
            input_ids += sep_encode
            labels += [IGNORE_INDEX] * len(sep_encode)


    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}



# Copyright (c) OpenMMLab. All rights reserved.
def template_map_fn(example, template):
    conversation = example.get("conversation", [])
    for i, single_turn_conversation in enumerate(conversation):
        input = single_turn_conversation.get("input", "")
        if input is None:
            input = ""
        input_text = template.INSTRUCTION.format(input=input, round=i + 1)
        system = single_turn_conversation.get("system", "")
        if system != "" and system is not None:
            system = template.SYSTEM.format(system=system)
            input_text = system + input_text
        single_turn_conversation["input"] = input_text

        if template.get("SUFFIX", None):
            output_text = single_turn_conversation.get("output", "")
            output_text += template.SUFFIX
            single_turn_conversation["output"] = output_text

        # SUFFIX_AS_EOS is False ==> need_eos_token is True
        single_turn_conversation["need_eos_token"] = not template.get(
            "SUFFIX_AS_EOS", False
        )
        single_turn_conversation["sep"] = template.get("SEP", "")

    return {"conversation": conversation}


def panorama_collect_fn(
        instances: Sequence[Dict],
        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
        return_hf_format: bool = False,
        use_varlen_attn: bool = False
):
    assert not return_hf_format, "return_hf_format is not supported yet."
    assert not use_varlen_attn, "use_varlen_attn is not supported yet."

    input_ids, labels = [], []

    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    has_pe = any(inst.get('image_grid_thw', None) is not None for inst in instances)
    has_grounding_image = any(inst.get('g_pixel_values') is not None for inst in instances)
    has_mask = any(inst.get('masks') is not None for inst in instances)


    has_vp = any(inst.get('vp_overall_mask') is not None for inst in instances)
    has_prompt_mask = any(inst.get('prompt_masks') is not None for inst in instances)
    assert has_vp and has_prompt_mask or not has_vp and not has_prompt_mask, \
    f"Inconsistent presence of visual prompts and prompt masks {has_vp} {has_prompt_mask}"

    pixel_values = []
    frames_per_batch = []

    image_grid_thw = []
    grounding_pixel_values = []
    object_masks = []
    seg_group_ids = []  # per-sample phrase index per mask (parallel to object_masks)
    vp_overall_mask = []
    prompt_masks = []
    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))

        if has_image:
            pixel_values.append(example['pixel_values'])
            if has_pe:
                image_grid_thw.append(example['image_grid_thw'])
            if has_vp:
                if 'vp_overall_mask' in example.keys() and example['vp_overall_mask'] is not None:
                    vp_overall_mask.append(example['vp_overall_mask'])
                else:
                    vp_overall_mask.append(torch.Tensor([False] * len(example['pixel_values'])))

        if has_grounding_image and 'g_pixel_values' in example.keys():
            if isinstance(example['g_pixel_values'], list):
                grounding_pixel_values += example['g_pixel_values']
                frames_per_batch.append(len(example['g_pixel_values']))
            else:
                grounding_pixel_values.append(example['g_pixel_values'])
                frames_per_batch.append(1)

        if has_mask:
            if 'masks' in example.keys() and example['masks'] is not None:
                if isinstance(example['masks'], list):
                    if isinstance(example['masks'][0], np.ndarray):
                        _masks = np.stack(example['masks'], axis=0)
                        _masks = torch.from_numpy(_masks)
                        object_masks.append(_masks)
                    else:
                        object_masks.append(torch.stack(example['masks'], dim=0))
                else:
                    object_masks.append(example['masks'])
                # parallel to object_masks; None for non-grouped samples
                seg_group_ids.append(example.get('seg_group_ids'))

        if has_prompt_mask:
            if 'prompt_masks' in example.keys():
                prompt_masks.append(example['prompt_masks'])

    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    # Some tokenizers have the same eos token and pad token, so input_ids
    # cannot be masked directly based on the pad token id.
    attention_mask = torch.zeros_like(input_ids).bool()
    for i, length in enumerate(ori_length):
        attention_mask[i, :length] = True

    bs, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'labels': labels
    }

    if has_image:
        data_dict['frames_per_batch'] = frames_per_batch
        data_dict['pixel_values'] = pixel_values
        for pixel_values_per_sample in pixel_values:
            assert isinstance(pixel_values_per_sample, torch.Tensor)
            # dim for internvl : [num_frames, 3, H, W]
            # dim for qwenvl : [L, C] C is 1176, L is length
            # assert pixel_values_per_sample.dim() == 4, "pixel_values must be a 4D tensor"

        if has_pe:
            data_dict['image_grid_thw'] = image_grid_thw

    if has_vp:
        data_dict['vp_overall_mask'] = torch.cat(vp_overall_mask, dim=0)

    if has_prompt_mask:
        data_dict['prompt_masks'] = prompt_masks

    if has_grounding_image:
        data_dict['g_pixel_values'] = grounding_pixel_values

    if has_mask:
        data_dict['masks'] = object_masks
        # phrase index per mask, when the dataset provides it
        if any(s is not None for s in seg_group_ids):
            data_dict['seg_group_ids'] = seg_group_ids

    return {'data': data_dict, 'data_samples': None}


class ConcatDatasetPanorama(TorchConcatDataset):

    def __init__(self, datasets:List[dict]):
        datasets_instance = []
        for cfg in datasets:
            datasets_instance.append(BUILDER.build(cfg))
        super().__init__(datasets=datasets_instance)

        print_log(
            f'Initialized ConcatDataset with {len(datasets)} datasets.'
        )
        for dataset in self.datasets:
            print_log(f'{repr(dataset.name)}')
            print_log(f'------Number of samples: {len(dataset)}')
            print_log(f'------Real Length: {dataset.real_len()}')

    def __repr__(self):
        main_str = 'Dataset as a concatenation of multiple datasets. \n'
        main_str += ',\n'.join(
            [f'{repr(dataset)}' for dataset in self.datasets])
        return main_str

