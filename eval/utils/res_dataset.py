# Referring-expression segmentation evaluation datasets (RefCOCO, RefCOCO+, RefCOCOg, gRefCOCO).
import os
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as _mask
from .refcoco_refer import REFER
from .grefcoco import G_REFER
from data.common import GRES_QUESTION
from .utils_refcoco import Summary, AverageMeter, intersectionAndUnionGPU
from .dist import master_only

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco+'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
    'grefcoco': {'splitBy': "unc", 'dataset_name': 'grefcoco'},
}

class RESDataset:
    def __init__(self,
                 image_folder,
                 dataset_name,
                 data_path=None,
                 split='val',
                 ):
        self.split = split
        self._set_attribute(dataset_name)
        json_datas = self.json_file_preprocess(data_path)
        self.json_datas = json_datas
        self.image_folder = image_folder

    def _set_attribute(self, dataset_name):
        attr_dict = DATASETS_ATTRIBUTES[dataset_name]
        self.splitBy = attr_dict['splitBy']
        self.dataset_name = attr_dict['dataset_name']

    def __len__(self):
        return len(self.json_datas)

    def json_file_preprocess(self, data_path):
        splitBy = self.splitBy
        dataset_name = self.dataset_name
        if dataset_name == "grefcoco":
            refer_api = G_REFER(data_path, dataset_name, splitBy)
        else:
            refer_api = REFER(data_path, dataset_name, splitBy)
        self.refer_api = refer_api
        ref_ids_train = refer_api.getRefIds(split=self.split)
        images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
        refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
        self.img2refs = self.create_img_to_refs_mapping(refs_train)

        image_infos = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        for item in loaded_images:
            item = item.copy()
            image_infos.append(item)

        self.annotations = refer_api.Anns
        refs = [self.img2refs[image_info['id']] for image_info in image_infos]

        ret = []
        for image_info, ref in zip(image_infos, refs):
            if len(ref) == 0:
                continue

            sents = []
            ann_ids = []
            for _ref in ref:
                for sent in _ref["sentences"]:
                    text = sent["sent"]
                    sents.append(text)
                    ann_ids.append(_ref["ann_id"])

            ret.append(
                {'image_info': image_info,
                 'sampled_ann_id': ann_ids,
                 'selected_labels': sents,
                 'image': image_info['file_name']
                 }
            )
        return ret

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def decode_mask(self, annotations_ids, image_info):
        masks = []

        for ann_id in annotations_ids:
            if isinstance(ann_id, list):
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = self.annotations[ann_id_i]
                        if self.dataset_name == "grefcoco":
                            if ann["iscrowd"]:
                                pass
                            else:
                                m = self.refer_api.getMask(ann)["mask"]
                                if m is not None:
                                    m_final = m_final | m
                            continue

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = _mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = _mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = self.annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = _mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = _mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)

        masks = torch.from_numpy(masks)
        return masks

    def only_get_text_infos(self, json_data):
        return {'sampled_sents': json_data['selected_labels']}

    def get_questions(self, text_require_infos):
        sampled_sents = text_require_infos['sampled_sents']
        # gRefCOCO uses its training prompt template; RefCOCO/+/g use the plain prompt.
        ret = []
        for sent in sampled_sents:
            if self.dataset_name == 'grefcoco':
                ret.append("<image>\n " + GRES_QUESTION.format(class_name=sent))
            else:
                ret.append("<image>\n Please segment {} in this image.".format(sent))

        return ret

    def filter_data_dict(self, data_dict):
        names = ['image', 'text', 'gt_masks', 'img_id', 'image_path']
        ret = {name: data_dict[name] for name in names}
        return ret

    def __getitem__(self, index):
        index = index % len(self)
        data_dict = self.json_datas[index]
        text_require_infos = self.only_get_text_infos(data_dict)
        questions = self.get_questions(text_require_infos)

        assert data_dict.get('image', None) is not None
        image_file = os.path.join(self.image_folder, data_dict['image'])
        image = Image.open(image_file).convert('RGB')

        # decode the GT masks for evaluation
        masks = self.decode_mask(data_dict['sampled_ann_id'], data_dict['image_info'])
        data_dict['gt_masks'] = masks
        data_dict['image'] = image
        data_dict['text'] = questions
        data_dict['img_id'] = str(index)
        data_dict['image_path'] = image_file
        return self.filter_data_dict(data_dict)

    @master_only
    def evaluate(self, result):
        # gRefCOCO has no-target expressions and uses the GRES metrics; RefCOCO/+/g use cIoU/gIoU.
        if self.dataset_name == 'grefcoco':
            return self._evaluate_gres(result)
        trackers = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
        }
        for pred_dict in result:
            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            masks = pred_dict['prediction_masks']
            _masks = []
            for mask in masks:
                if mask is not None:
                    mask = rle_to_mask(mask)
                _masks.append(mask)
            targets = pred_dict['gt_masks']
            _targets = rle_to_mask(targets)

            for i_item, _mask in enumerate(_masks):
                _target = _targets[i_item: i_item+1]
                if _mask is None:
                    _mask = _target * 0
                for prediction, target in zip(_mask, _target):
                    prediction = torch.from_numpy(prediction).int().cuda()
                    target = torch.from_numpy(target).int().cuda()
                    intersect, union_, _ = intersectionAndUnionGPU(
                        prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                    )
                    intersection += intersect
                    union += union_
                    accuracy_iou += intersect / (union_ + 1e-5)
                    accuracy_iou[union_ == 0] += 1.0

            if not isinstance(intersection, float):
                intersection = intersection.cpu().numpy()
            if not isinstance(union, float):
                union = union.cpu().numpy()
            if not isinstance(accuracy_iou, float):
                accuracy_iou = accuracy_iou.cpu().numpy()
            accuracy_iou = accuracy_iou / _targets.shape[0]
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=_targets.shape[0])

        cur_results = {'pixel_intersection': trackers["intersection"].sum[1],
                       'pixel_union': trackers["union"].sum[1],
                       'gIoU': trackers["gIoU"].avg[1],
                       'mask_counts': trackers["gIoU"].count,
                       }
        class_iou = cur_results['pixel_intersection'] / (cur_results['pixel_union'] + 1e-10)
        global_iou = cur_results['gIoU']

        print('============================================')
        print('CIoU: {}, GIoU: {}'.format(class_iou, global_iou))
        print('============================================')
        print('RES_{}_{} successfully finished evaluating'.format(self.dataset_name, self.split))
        return {'Acc': class_iou}

    def _evaluate_gres(self, result):
        """GRES metrics (cIoU, gIoU, N-acc, T-acc): per expression, a no-target GT counts as
        TP/FN by whether the prediction is empty, and a targeted GT as TN/FP; gIoU is 1 for a
        correct empty prediction and 0 for a wrong one. Returns raw fractions in [0, 1].
        """
        inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        g_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        nt_tp = nt_tn = nt_fp = nt_fn = 0.0
        for pred_dict in result:
            masks = pred_dict['prediction_masks']
            _masks = [rle_to_mask(m) if m is not None else None for m in masks]
            _targets = rle_to_mask(pred_dict['gt_masks'])
            for i_item in range(_targets.shape[0]):
                gt = (_targets[i_item] > 0).astype(np.uint8)
                _m = _masks[i_item] if i_item < len(_masks) else None
                pred = (np.zeros_like(gt) if _m is None
                        else (_m.sum(0) > 0).astype(np.uint8))
                gt_t = torch.from_numpy(gt).int().cuda()
                pred_t = torch.from_numpy(pred).int().cuda()
                if gt_t.sum() < 1.0:                                   # no-target GT
                    if pred_t.sum() < 1.0:
                        nt_tp += 1.0
                        g_iou_meter.update(1.0)
                    else:
                        _, union_i, _ = intersectionAndUnionGPU(
                            pred_t.contiguous().clone(), gt_t.contiguous().clone(), 2, ignore_index=255)
                        nt_fn += 1.0
                        g_iou_meter.update(0.0)
                        union_meter.update(union_i.cpu().numpy())
                else:                                                 # targeted GT
                    if pred_t.sum() < 1.0:
                        nt_fp += 1.0
                    else:
                        nt_tn += 1.0
                    inter_i, union_i, _ = intersectionAndUnionGPU(
                        pred_t.contiguous().clone(), gt_t.contiguous().clone(), 2, ignore_index=255)
                    inter_i = inter_i.cpu().numpy()
                    union_i = union_i.cpu().numpy()
                    this_giou = inter_i / (union_i + 1e-8)
                    inter_meter.update(inter_i)
                    union_meter.update(union_i)
                    g_iou_meter.update(this_giou)

        n_acc = nt_tp / (nt_tp + nt_fn + 1e-10)
        t_acc = nt_tn / (nt_tn + nt_fp + 1e-10)
        g_iou = float(g_iou_meter.avg[1])
        c_iou = float((inter_meter.sum / (union_meter.sum + 1e-10))[1])

        print('============================================')
        print('GRES {}/{}  cIoU: {:.2f}  gIoU: {:.2f}  N-acc: {:.2f}  T-acc: {:.2f}'.format(
            self.dataset_name, self.split, c_iou * 100, g_iou * 100, n_acc * 100, t_acc * 100))
        print('  (NT_TP={:.0f} NT_FN={:.0f} NT_FP={:.0f} NT_TN={:.0f})'.format(
            nt_tp, nt_fn, nt_fp, nt_tn))
        print('============================================')
        print('RES_{}_{} successfully finished evaluating'.format(self.dataset_name, self.split))
        return {'c_iou': c_iou, 'g_iou': g_iou, 'N_acc': n_acc, 'T_acc': t_acc,
                'nt_tp': nt_tp, 'nt_fn': nt_fn, 'nt_fp': nt_fp, 'nt_tn': nt_tn}


def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = _mask.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask
