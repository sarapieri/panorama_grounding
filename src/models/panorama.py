"""PANORAMA model: the VLM writes a grounded caption with one [SEG] per phrase; the projected [SEG]
conditions SAM 3's proposals and selects among them."""
import torch
import torch.nn.functional as F

from .panorama_base import PanoramaBase


class Panorama(PanoramaBase):

    # Trainable SAM 3 parts for unfreeze_concept; the vision backbone always stays frozen.
    _UNFREEZE_TABLE = {
        'fusion':     'transformer.encoder',   # fusion encoder (concept x image conditioning)
        'decoder':    'transformer.decoder',   # DETR decoder (object queries)
        'maskformer': 'segmentation_head',     # MaskFormer pixel decoder + heads
        'scorer':     'dot_prod_scoring',      # per-query match scorer
    }

    def __init__(self, *args,
                 match_score_lambda=0.0, semantic_loss_weight=0.0,
                 unfreeze_concept=None,
                 score_loss_weight: float = 2.0, focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0, score_threshold: float = 0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._apply_concept_unfreeze(unfreeze_concept)
        self._log_whole_model_summary()

        self.score_loss_weight = score_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.score_threshold = score_threshold

        assert match_score_lambda >= 0.0, match_score_lambda
        self.match_score_lambda = float(match_score_lambda)
        if self.match_score_lambda > 0:
            print(f"[PANORAMA] match_score_lambda={self.match_score_lambda}: score-aware "
                  f"Hungarian (cost = dice - lambda*sigmoid(seg_logit); assignment only, "
                  f"never a loss term).")
        assert semantic_loss_weight >= 0.0, semantic_loss_weight
        self.semantic_loss_weight = float(semantic_loss_weight)
        if self.semantic_loss_weight > 0:
            print(f"[PANORAMA] semantic_loss_weight={self.semantic_loss_weight}: semantic loss "
                  f"on the per-phrase GT union.")
        print("[PANORAMA] fusion prompt = the projected [SEG] alone; the same token scores the "
              "proposals.")

    def state_dict(self, *args, **kwargs):
        """Also save the trainable proposal-model params (unfreeze_concept); the base saves only
        the VLM and the [SEG] bridge."""
        prefix = kwargs.get('prefix', '')
        sd = super().state_dict(*args, **kwargs)
        ge_prefix = prefix + 'grounding_encoder.'
        ge_sd = self.grounding_encoder.state_dict(prefix=ge_prefix)
        trainable = {ge_prefix + n for n, p in self.grounding_encoder.named_parameters()
                     if p.requires_grad}
        sd.update({k: v for k, v in ge_sd.items() if k in trainable})
        return sd

    def _apply_concept_unfreeze(self, names):
        """After the base __init__ froze the whole proposal model, unfreeze the named submodules
        (e.g. ['fusion', 'scorer']). The vision backbone is never unfrozen. None -> everything
        in the proposal model stays frozen."""
        img = self.grounding_encoder.image_model
        if not names:
            return
        for n in names:
            attr = self._UNFREEZE_TABLE.get(n)
            assert attr is not None, f"unknown unfreeze target '{n}'; valid: {list(self._UNFREEZE_TABLE)}"
            mod = img
            for p in attr.split('.'):
                mod = getattr(mod, p)
            mod.requires_grad_(True)
            n_tr = sum(p.numel() for p in mod.parameters())
            print(f"[PANORAMA] unfroze '{n}' ({attr}): {n_tr/1e6:.2f}M params -> trainable")
        ge = self.grounding_encoder
        ge_tr = sum(p.numel() for p in ge.parameters() if p.requires_grad)
        ge_tot = sum(p.numel() for p in ge.parameters())
        print(f"[PANORAMA] grounding_encoder trainable: {ge_tr/1e6:.2f}M / {ge_tot/1e6:.2f}M "
              f"(unfrozen: {names}; vision backbone stays frozen)")

    def _sigmoid_focal(self, logits, targets):
        """Summed sigmoid focal loss over a (Q,) query vector (caller normalizes by #positives)."""
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce * (1 - p_t) ** self.focal_gamma
        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.sum()

    @torch.no_grad()
    def _match(self, q_masks, gt_set, score_bonus=None):
        """Hungarian match by soft-dice cost; score_bonus breaks ties between near-duplicate masks."""
        from scipy.optimize import linear_sum_assignment
        q = q_masks.flatten(1).sigmoid()
        g = (gt_set.flatten(1) > 0.5).float()
        num = g @ q.t()
        denom = g.sum(1)[:, None] + q.sum(1)[None, :]
        dice = 2.0 * num / denom.clamp(min=1.0)
        cost = 1.0 - dice
        if score_bonus is not None:
            cost = cost - score_bonus[None, :].to(cost)
        cost = cost.cpu().numpy()
        row, col = linear_sum_assignment(cost)
        order = col[row.argsort()]
        return torch.as_tensor(order, device=q_masks.device, dtype=torch.long)

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        seg_group_ids = data.pop('seg_group_ids', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pseudo_data(
                dtype=self.torch_dtype, device=input_ids.device)
            seg_group_ids = None
        else:
            seg_valid = True

        device = input_ids.device
        B = len(gt_masks)

        # Project every position (keeps the graph connected); the [SEG] rows are the concept and
        # the scorer key.
        proj = self.text_hidden_fcs(output.hidden_states[-1])
        _zero = proj.mean() * 0.0
        if seg_valid:
            seg_pos = input_ids == self.seg_token_idx
            counts = seg_pos.sum(-1).tolist()
            seg_embs = proj[seg_pos]
        else:
            counts = [int(g.shape[0]) for g in gt_masks]
            seg_embs = torch.cat([proj[i, :n] for i, n in enumerate(counts)], 0)

        # Attach each phrase's GT masks and cap phrases per image.
        sel_embs, phrase_gt, img_ids_list = [], [], []
        off = 0
        for i in range(B):
            n_ph = counts[i]
            emb_i = seg_embs[off:off + n_ph]
            off += n_ph
            gt_i = gt_masks[i]
            if gt_i is None or gt_i.shape[0] == 0:
                continue
            if seg_group_ids is not None and seg_group_ids[i] is not None:
                grp_i = seg_group_ids[i].to(device)
            else:
                grp_i = torch.arange(gt_i.shape[0], device=device)
            n_eff = min(n_ph, int(grp_i.max().item()) + 1) if grp_i.numel() else 0
            phrase_ids = list(range(n_eff))
            if len(phrase_ids) > self.max_objs_per_image:
                keep = torch.randperm(len(phrase_ids))[:self.max_objs_per_image].tolist()
                phrase_ids = [phrase_ids[k] for k in keep]
            for j in phrase_ids:
                gset = gt_i[grp_i == j]
                if gset.shape[0] == 0:
                    continue
                sel_embs.append(emb_i[j])
                phrase_gt.append(gset)
                img_ids_list.append(i)

        if len(sel_embs) == 0:
            z = _zero + output.loss * 0.0
            # Touch every trainable grounding param with a zero term so all ranks produce the same
            # gradient set; otherwise the all-reduce waits forever on ranks that skip SAM 3.
            for _p in self.grounding_encoder.parameters():
                if _p.requires_grad:
                    z = z + _p.sum() * 0.0
            if not getattr(self, '_no_target_warned', False):
                self._no_target_warned = True
                print('[PANORAMA] no supervisable phrase in this micro-batch -> zero grounding '
                      'loss with a rank-consistent gradient set (logged once per process)')
            ret = {'loss_mask': z, 'loss_dice': z, 'loss_score': z,
                   'llm_loss': output.loss, 'mask_iou': output.loss.new_zeros(())}
            if self.semantic_loss_weight > 0:  # keep loss keys consistent across steps
                ret['loss_sem_mask'] = z
                ret['loss_sem_dice'] = z
            return ret

        img_ids = torch.as_tensor(img_ids_list, device=device)
        images = torch.stack([self.grounding_encoder.preprocess_image(p) for p in g_pixel_values])
        images = images.to(self.torch_dtype)

        # SAM 3 proposals conditioned on the projected [SEG].
        seg_prompt = torch.stack(sel_embs, 0) + _zero
        out = self.grounding_encoder.forward_concept(images, seg_prompt, img_ids=img_ids)
        pred_masks = out['pred_masks'].float()

        # Score the queries with SAM 3's scorer, keyed by the same [SEG].
        queries = out['queries']
        dps = self.grounding_encoder.image_model.dot_prod_scoring
        # autocast needed: queries are fp32, scorer weights bf16.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            seg_logits = dps(
                queries.unsqueeze(0),
                seg_prompt.unsqueeze(0).to(queries.dtype),
                torch.zeros(len(sel_embs), 1, dtype=torch.bool, device=device),
            )[0].squeeze(-1).float()
        hh, ww = pred_masks.shape[-2:]
        Q = pred_masks.shape[1]

        loss_mask = pred_masks.new_zeros(())
        loss_dice = pred_masks.new_zeros(())
        loss_score = seg_logits.new_zeros(())
        loss_sem_mask = pred_masks.new_zeros(())
        loss_sem_dice = pred_masks.new_zeros(())
        n_pos = 0
        ious = []
        for p in range(len(phrase_gt)):
            gset = F.interpolate(phrase_gt[p].unsqueeze(0).float(), size=(hh, ww),
                                 mode='nearest').squeeze(0).to(device)
            if gset.shape[0] > Q:  # Hungarian needs n <= Q
                gset = gset[:Q]
            qmasks = pred_masks[p]
            _bonus = ((self.match_score_lambda * seg_logits[p].sigmoid()).detach()
                      if self.match_score_lambda > 0 else None)
            match = self._match(qmasks, gset, score_bonus=_bonus)
            mp = qmasks[match]
            if self.loss_sample_points:
                sp, sg = self.sample_points(mp, gset)
                loss_dice = loss_dice + self.loss_dice(sp, sg, avg_factor=(gset.shape[0] + 1e-4))
                loss_mask = loss_mask + self.loss_mask(
                    sp.reshape(-1), sg.reshape(-1),
                    avg_factor=(mp.shape[0] * sp.shape[1] + 1e-4))
            else:
                loss_mask = loss_mask + self.loss_mask(mp, gset)
                loss_dice = loss_dice + self.loss_dice(mp, gset)
            # One positive per matched query; the rest are negatives.
            tgt = torch.zeros(Q, device=device)
            tgt[match] = 1.0
            loss_score = loss_score + self._sigmoid_focal(seg_logits[p], tgt)
            n_pos += gset.shape[0]
            if self.semantic_loss_weight > 0:
                # Semantic head vs the phrase's GT union.
                sem_p = out['semantic_seg'][p].float()
                union_gt = gset.max(dim=0, keepdim=True).values
                if self.loss_sample_points:
                    ssp, ssg = self.sample_points(sem_p, union_gt)
                    loss_sem_dice = loss_sem_dice + self.loss_dice(ssp, ssg, avg_factor=(1 + 1e-4))
                    loss_sem_mask = loss_sem_mask + self.loss_mask(
                        ssp.reshape(-1), ssg.reshape(-1), avg_factor=(ssp.shape[1] + 1e-4))
                else:
                    loss_sem_mask = loss_sem_mask + self.loss_mask(sem_p, union_gt)
                    loss_sem_dice = loss_sem_dice + self.loss_dice(sem_p, union_gt)
            with torch.no_grad():
                pm, gm = (mp > 0), (gset > 0.5)
                inter = (pm & gm).flatten(1).sum(-1).float()
                union = (pm | gm).flatten(1).sum(-1).float()
                ious.append((inter / union.clamp(min=1.0)).mean())

        n_ph = max(len(phrase_gt), 1)
        loss_mask = loss_mask / n_ph
        loss_dice = loss_dice / n_ph
        loss_score = (loss_score / max(n_pos, 1)) * self.score_loss_weight
        mask_iou = torch.stack(ious).mean() if ious else pred_masks.new_zeros(())

        _scale = 1.0 if seg_valid else 0.0
        ret = {
            'loss_mask': loss_mask * _scale,
            'loss_dice': loss_dice * _scale,
            'loss_score': loss_score * _scale,
            'llm_loss': output.loss,
            'mask_iou': mask_iou,
        }
        if self.semantic_loss_weight > 0:
            _sw = self.semantic_loss_weight / n_ph
            ret['loss_sem_mask'] = loss_sem_mask * _sw * _scale
            ret['loss_sem_dice'] = loss_sem_dice * _sw * _scale
        return ret
