import re

from .dist import (
    _init_dist_pytorch,
    _init_dist_slurm,
    get_dist_info,
    is_distributed,
    get_rank,
    barrier,
    collect_results_cpu,
)

from .rle import mask_to_rle_pytorch, coco_encode_rle, mask_to_coco_rle
from .text_similarity import TextSimilarityMetric

from .vl_metrics import (
    build_dedup_views,
    evaluate_mask_miou,
    evaluate_with_mapping,
)


def find_seg_indices(text):
    """Indices (in order of appearance) of the [SEG] tokens in a generated answer."""
    return list(range(len(re.findall(r'\[SEG\]', text))))


__all__ = [
    # dist helpers
    "_init_dist_pytorch",
    "_init_dist_slurm",
    "get_dist_info",
    "is_distributed",
    "get_rank",
    "barrier",
    "collect_results_cpu",
    # text similarity
    "TextSimilarityMetric",
    # [SEG] parsing
    "find_seg_indices",
    # RLE encoding
    "mask_to_rle_pytorch",
    "coco_encode_rle",
    "mask_to_coco_rle",
    # VL / mask metrics
    "build_dedup_views",
    "evaluate_mask_miou",
    "evaluate_with_mapping",
]
