# Mask -> COCO RLE encoding shared by the eval launchers.
import numpy as np
import torch
from pycocotools import mask as mask_utils


def mask_to_rle_pytorch(tensor: torch.Tensor):
    """Encode a (B, H, W) binary tensor into uncompressed RLEs in the pycocotools format."""
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)  # fortran order, flattened h*w

    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out


def coco_encode_rle(uncompressed_rle):
    """Compress an uncompressed RLE and make it json-serializable."""
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_coco_rle(mask):
    """(N, H, W) array -> list of pycocotools RLE dicts with string counts (json-serializable)."""
    rle = []
    for m in mask:
        rle.append(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle
