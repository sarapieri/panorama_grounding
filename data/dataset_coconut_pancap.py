# COCONut-PanCap-Recaptioned: our regenerated grounded captions for the COCONut-PanCap images.
# Ours: released at https://huggingface.co/datasets/Panorama-grounding/COCONut-PanCap-Recaptioned
# Sources: COCONut-PanCap (Deng et al., 2025, https://arxiv.org/abs/2502.02589) and
#          COCONut (Deng et al., CVPR 2024, https://github.com/bytedance/coconut_cvpr2024).
# Loader: per-image caption .txt + COCONut-S panoptic PNG/json -> one '<p> phrase </p> [SEG]'
# per phrase, with the phrase's mask set grouped through seg_group_ids.
import copy
import hashlib
import json
import os
import pickle
import re
import random
from typing import Literal, Dict, List, Any, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from .base import PanoramaBaseDataset
from .common import PANOCAPS_QUESTIONS
from .dataset_panocaps import panocaps_tag_caption


class COCONutPanCapDataset(PanoramaBaseDataset):
    """COCONut-PanCap-Recaptioned: captions tagged with panoptic segment ids -> grounded
    captioning samples (segment ids are looked up in the panoptic json's segments_info)."""

    def __init__(self,
                 data_path: str,
                 image_folder: str,
                 masks_dir: str = None,
                 panoptic_json: str = None,
                 tokenizer=None,
                 prompt_template=None,
                 max_length: int = 2048,
                 special_tokens=None,
                 arch_type: Literal['qwen'] = 'qwen',
                 preprocessor=None,
                 extra_image_processor=None,
                 cache_dir: str = None,
                 use_cache: bool = True,
                 repeats: float = 1.0,
                 name: str = 'COCONutPanCapDataset',
                 **kwargs):
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name,
            **kwargs,
        )

        self.data_path = data_path
        self.image_folder = image_folder
        self.question_templates = PANOCAPS_QUESTIONS
        self.begin_str = '<image>\n'

        # COCONut-PanCap-Recaptioned layout: caption_train2017/ (one .txt per image) and the
        # COCONut-S panoptic json next to it; the COCONut-S panoptic PNGs live under masks_dir.
        self.captions_dir = os.path.join(data_path, 'caption_train2017')
        self.masks_dir = masks_dir if masks_dir is not None \
            else os.path.join(data_path, 'coconut_s', 'panoptic')
        self.panoptic_json_file = panoptic_json if panoptic_json is not None \
            else os.path.join(data_path, 'coconut_s_panoptic_train2017.json')

        # Manifest cache: building the sample list walks ~118k caption files, checks every
        # image and mask file and parses the large panoptic json, so the result is cached and
        # reused while the inputs are unchanged. Default location: data/.cache (regenerable).
        # Override via cache_dir or PANORAMA_CACHE_DIR.
        self.use_cache = use_cache
        if cache_dir is not None:
            self.cache_dir = cache_dir
        elif os.environ.get('PANORAMA_CACHE_DIR'):
            self.cache_dir = os.path.join(os.environ['PANORAMA_CACHE_DIR'], 'coconut_cache')
        else:
            self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')

        self.data_list = self._load_annotations()
        if not self.data_list:
            raise RuntimeError(f"No COCONut-PanCap samples found under '{self.data_path}'.")

    def _load_annotations(self) -> List[Dict]:
        """Build the sample list (caption .txt + panoptic PNG), with a manifest cache.

        Under DDP only rank 0 builds and writes the cache; the other ranks wait on a barrier
        and read it, so the file-system walk happens once instead of once per rank.
        """
        assert (os.path.isdir(self.captions_dir)
                and os.path.isdir(self.masks_dir)
                and os.path.isfile(self.panoptic_json_file)), (
            f"Invalid COCONut-PanCap paths:\n  captions: {self.captions_dir}\n"
            f"  masks: {self.masks_dir}\n  panoptic json: {self.panoptic_json_file}")

        print(f"Loading COCONut-PanCap captions: {self.captions_dir}\n  masks: {self.masks_dir}")

        # one readdir (cheap), used both for the cache fingerprint and for building
        cap_files = sorted(f for f in os.listdir(self.captions_dir) if f.endswith(".txt"))
        meta = self._manifest_meta(cap_files)
        cache_path = self._manifest_cache_path()

        dist_on = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if dist_on else 0

        data_list = None
        if rank == 0:
            if self.use_cache:
                data_list = self._load_manifest(cache_path, meta)
            if data_list is None:
                data_list = self._build_data_list(cap_files)
                if self.use_cache:
                    self._save_manifest(cache_path, data_list, meta)

        if dist_on:
            dist.barrier()  # rank 0 has built/cached; the rest now read it
            if rank != 0:
                data_list = self._load_manifest(cache_path, meta)
                if data_list is None:  # cache unreadable on this rank: build locally
                    data_list = self._build_data_list(cap_files)

        return data_list

    def _manifest_meta(self, cap_files: List[str]) -> Dict:
        """Cheap fingerprint of the inputs; any mismatch invalidates the cache."""
        return {
            "version": 1,
            "captions_dir": self.captions_dir,
            "masks_dir": self.masks_dir,
            "panoptic_json_file": self.panoptic_json_file,
            "image_folder": self.image_folder,
            "panoptic_json_size": os.path.getsize(self.panoptic_json_file),
            "panoptic_json_mtime": int(os.path.getmtime(self.panoptic_json_file)),
            "n_caption_files": len(cap_files),
        }

    def _manifest_cache_path(self) -> str:
        key = hashlib.md5("|".join([
            self.captions_dir, self.masks_dir,
            self.panoptic_json_file, self.image_folder,
        ]).encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"coconut_manifest_{key}.pkl")

    def _load_manifest(self, cache_path: str, meta: Dict):
        """Return the cached data_list if present and still valid, else None."""
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
        except Exception as e:
            print(f"[COCONut][WARN] could not read manifest cache ({e}); rebuilding")
            return None
        if blob.get("meta") != meta:
            print("[COCONut] manifest cache stale (inputs changed); rebuilding")
            return None
        data = blob.get("data") or []
        print(f"[COCONut] Loaded {len(data)} samples from manifest cache: {cache_path}")
        return data

    def _save_manifest(self, cache_path: str, data_list: List[Dict], meta: Dict):
        tmp = None
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump({"meta": meta, "data": data_list}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)  # atomic
            print(f"[COCONut] Wrote manifest cache: {cache_path}")
        except Exception as e:
            print(f"[COCONut][WARN] failed to write manifest cache ({e}); continuing")
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _build_data_list(self, cap_files: List[str]) -> List[Dict]:
        """Slow path: parse the panoptic json + walk all caption files."""
        # panoptic json gives the segments_info (segment ids) per mask file
        with open(self.panoptic_json_file, "r", encoding="utf-8") as f:
            pan = json.load(f)
        ann_by_file = {a["file_name"]: a["segments_info"] for a in pan.get("annotations", [])}

        data_list = []
        n_total, n_bad = 0, 0
        for f in cap_files:
            image_id = os.path.splitext(f)[0]
            cap_path = os.path.join(self.captions_dir, image_id + ".txt")
            mask_path = os.path.join(self.masks_dir, image_id + ".png")
            image_path = os.path.join(self.image_folder, image_id + ".jpg")

            if not os.path.isfile(mask_path):
                print(f"[COCONut][WARN] missing mask file: {mask_path}")
                continue
            if not os.path.isfile(image_path):
                print(f"[COCONut][WARN] missing image file: {image_path}")
                continue

            with open(cap_path, "r", encoding="utf-8") as cf:
                caption = cf.read().strip()

            # parse the tagged caption to get the mask order; skip malformed ones
            _, gt_mask_order, _, caption_ok = self.parse_caption_text(caption)
            n_total += 1
            if not caption_ok:
                n_bad += 1
                continue

            data_list.append({
                "caption": caption,
                "gt_mask_order": gt_mask_order,
                "mask_path": mask_path,
                "image": image_path,
                "file_name": image_id,
                "segments_info": ann_by_file.get(image_id + ".png", []),
            })

        print(f"[COCONut] Loaded {n_total} captions -- {n_bad} not parsable correctly.")
        return data_list

    @staticmethod
    def parse_caption_text(text: str) -> Tuple[List[str], List[int], List[List[int]], bool]:
        """Parse caption tags like '<4:large zebu>' or '<6,7,8:eight boys>'.

        Returns (labels, gt_mask_order, mask_groups, ok). ``ok`` is False if the
        caption is malformed (some '<...>' block does not match the tag pattern,
        or a tag has no ids / no description) or contains no tags at all.
        """
        TAG_RE = re.compile(r"<\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*:\s*([^>]*)>")

        labels: List[str] = []
        gt_mask_order: List[int] = []
        mask_groups: List[List[int]] = []

        all_blocks = re.findall(r"<[^>]*>", text)
        matches = list(TAG_RE.finditer(text))
        matched_blocks = [m.group(0) for m in matches]

        # malformed if there are no tags, or some '<...>' block doesn't match
        ok = (len(all_blocks) > 0) and (set(all_blocks) == set(matched_blocks))

        for m in matches:
            idx_part = m.group(1)
            desc = (m.group(2) or "").strip()

            group = [int(tok.strip()) for tok in idx_part.split(",") if tok.strip().isdigit()]
            if not group or not desc:
                ok = False
                continue

            labels.append(desc)
            mask_groups.append(group)
            gt_mask_order.extend(group)

        return labels, gt_mask_order, mask_groups, ok

    def decode_id_mask_png(self, mask_path: str, segments_info: List[Dict]):
        """Decode a panoptic PNG into (N, H, W) uint8 masks.

        Masks follow the segments_info order, filtered to the segment ids that
        actually appear in the PNG. Returns (masks_tensor, kept_ids) or
        (None, None) if no segment is present.
        """
        arr = np.array(Image.open(mask_path))
        if arr.ndim == 3:
            arr = arr[..., 0]

        existing_ids = set(np.unique(arr).tolist())
        existing_ids.discard(0)

        kept_ids = [int(s["id"]) for s in segments_info if int(s["id"]) in existing_ids]
        if len(kept_ids) == 0:
            return None, None

        masks = np.stack([(arr == sid).astype(np.uint8) for sid in kept_ids], axis=0)
        return torch.from_numpy(masks), kept_ids

    def real_len(self) -> int:
        return len(self.data_list)


    def _create_conversation(self, caption: str) -> List[Dict]:
        """Build the interleaved grounded-captioning conversation."""
        question = random.choice(self.question_templates).strip()
        question = self.begin_str + question
        answer = panocaps_tag_caption(caption)
        return [
            {'from': 'human', 'value': question},
            {'from': 'gpt', 'value': answer},
        ]

    def prepare_data(self, index: int) -> Dict[str, Any]:
        data_dict = copy.deepcopy(self.data_list[index])

        # read image
        image = self._read_image(data_dict['image'])
        if image is None:
            return None

        # decode panoptic-PNG masks (segments_info order, present ids only)
        decoded_masks, kept_ids = self.decode_id_mask_png(
            data_dict['mask_path'], data_dict['segments_info']
        )
        if decoded_masks is None or decoded_masks.shape[0] == 0:
            return None
        n_masks = decoded_masks.shape[0]

        # caption ids are segment ids -> map to positions in decoded_masks.
        # 0 is background and therefore invalid. Skip if any id is missing.
        gt_order = np.asarray(data_dict['gt_mask_order'], dtype=int)
        if gt_order.size == 0:
            return None
        id2pos = {int(sid): i for i, sid in enumerate(kept_ids)}
        pos = np.array([id2pos.get(int(x), -1) if int(x) != 0 else -1 for x in gt_order], dtype=int)
        if (pos < 0).any() or (pos >= n_masks).any():
            print(f"[COCONut][WARN] {data_dict['file_name']} caption has invalid segment ids "
                  f"(num masks: {n_masks})")
            return None

        # reorder GT masks to the [SEG] order of the caption
        masks = decoded_masks.index_select(0, torch.as_tensor(pos, dtype=torch.long))

        # Build the conversation (one '<p> phrase </p> [SEG]' per phrase) and the phrase index
        # of every mask (a phrase may ground to several masks, e.g. '<6,7,8:eight boys>').
        # Skip the sample if the phrase count and the mask grouping disagree.
        conversation = self._create_conversation(data_dict['caption'])
        _, _, mask_groups, _ = self.parse_caption_text(data_dict['caption'])
        grp = np.asarray([j for j, g in enumerate(mask_groups) for _ in g], dtype=int)
        n_phrases = (int(grp.max()) + 1) if grp.size else 0
        n_tags = conversation[1]['value'].count('</p>')
        if grp.size != masks.shape[0] or n_tags != n_phrases:
            print(f"[COCONut][WARN] {data_dict['file_name']} grouped phrase mismatch: "
                  f"{n_tags} <p> tags vs {n_phrases} phrases ({grp.size} grp ids vs {masks.shape[0]} masks)")
            return None
        out_data_dict = {'masks': masks,
                         'seg_group_ids': torch.as_tensor(grp, dtype=torch.long)}

        # process image (pixel_values, image_grid_thw, g_pixel_values, num_image_tokens)
        image_data = self._process_single_image(image)
        out_data_dict.update(image_data)

        # encode the conversation
        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(conversation, image_token_str)
        token_dict = self.get_inputid_labels(conversation)
        out_data_dict.update(token_dict)

        return out_data_dict
