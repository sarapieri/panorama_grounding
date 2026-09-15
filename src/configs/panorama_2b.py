# PANORAMA-2B: Qwen3-VL-2B-Instruct + SAM 3, trained on the full mixture.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3VLProcessor

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE


from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from src.models import Panorama, Sam3ConceptTrainRunner, DirectResize
from data import (
    panorama_collect_fn, PanoramaRefSeg, PanoramaGCGDataset, PanoCapsDataset,
    COCONutPanCapDataset, PanoramaGRefSeg, PanoramaPhraseCut
)

from data.data_utils import ConcatDatasetPanorama
from data.dataset_mrseg_single_turn import PanoramaMRSegSingleTurn
from src.models.qwen3vl import Qwen3VL

#######################################################################
#                          PART 1  Settings                           #
#######################################################################


def _load_env(key, default):
    """Read KEY from the repo-root .env. Builtins only: mmengine's lazy config parser forbids
    calling functions on imported modules such as ``os``."""
    try:
        with open('.env') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and \
                        line.split('=', 1)[0].strip() == key:
                    return line.split('=', 1)[1].strip()
    except OSError:
        pass
    return default


# Model
path = 'Qwen/Qwen3-VL-2B-Instruct'
pretrained_pth = None

# SAM 3 input size (square).
sam3_img_size = 1008
# SAM 3 checkpoint; PANORAMA_SAM3_PATH in .env.
sam3_ckpt = _load_env('PANORAMA_SAM3_PATH', 'sam3.pt')

# Data
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

# Scheduler & Optimizer
# Effective batch 128 = batch_size x accumulative_counts x 16 GPUs.
batch_size = 4
accumulative_counts = 2
dataloader_num_workers = 16
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 2000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ['[SEG]', '<p>', '</p>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=sam3_img_size,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=Panorama,
    training_bs=batch_size,
    grounding_img_size=sam3_img_size,
    max_objs_per_image=10,  # max phrases supervised per image
    score_loss_weight=0.5,  # weight of the score focal loss
    score_threshold=0.5,    # inference threshold on the match score
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    match_score_lambda=0.25,        # score-aware Hungarian: cost = dice - 0.25*sigmoid(seg_logit)
    semantic_loss_weight=1.0,       # mask+dice also on the semantic head vs the per-phrase GT union
    unfreeze_concept=['fusion', 'scorer'],  # +10.7M trainable in SAM 3: fusion encoder + match scorer
    arch_type='qwen',
    mllm=dict(
        type=Qwen3VL,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            modules_to_save=['lm_head', 'embed_tokens'],
            target_modules=None,
        ),
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=Sam3ConceptTrainRunner,
        ckpt_path=sam3_ckpt,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5)
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

# Data root: override via PANORAMA_DATA_ROOT in .env.
DATA_ROOT = _load_env('PANORAMA_DATA_ROOT', './data/')

panorama_default_dataset_configs=dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    max_length=max_length,
    arch_type='qwen',
    preprocessor=dict(
        type=Qwen3VLProcessor.from_pretrained,
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
    )
)

######################### ImageRefSeg #################################
RES_ROOT = DATA_ROOT + 'ref_seg/'
panorama_data_refseg_configs = [
    dict(
        type=PanoramaRefSeg,
        name='RefCOCO',
        data_root=RES_ROOT + 'refcoco',
        data_prefix=dict(img_path='coco2014/train2014/'),
        ann_file='instances.json',
        split_file='refs(unc).p',
        num_classes_per_sample=5,
        repeats=5,
        **panorama_default_dataset_configs,
    ),
    dict(
        type=PanoramaRefSeg,
        name='RefCOCO+',
        data_root=RES_ROOT + 'refcoco+',
        data_prefix=dict(img_path='coco2014/train2014/'),
        ann_file='instances.json',
        split_file='refs(unc).p',
        num_classes_per_sample=5,
        repeats=5,
        **panorama_default_dataset_configs,
    ),
    dict(
        type=PanoramaRefSeg,
        name='RefCOCOg',
        data_root=RES_ROOT + 'refcocog',
        data_prefix=dict(img_path='coco2014/train2014/'),
        ann_file='instances.json',
        split_file='refs(umd).p',
        num_classes_per_sample=5,
        repeats=4,
        **panorama_default_dataset_configs,
    ),
]


######################### GCG #########################################
panorama_data_gcg_configs = [
    dict(
        type=PanoramaGCGDataset,
        name='GCG_RefCOCOg',
        image_folder=DATA_ROOT + 'glamm_data/images/coco2014/train2014/',
        data_path=DATA_ROOT + 'glamm_data/annotations/RefCOCOg_GCG_train.json',
        dataset_type='refcocog',
        repeats=3,
        **panorama_default_dataset_configs
    ),
    dict(
        type=PanoramaGCGDataset,
        name='GCG_GranDf',
        image_folder=DATA_ROOT + 'glamm_data/images/grandf/train/',
        data_path=DATA_ROOT + 'glamm_data/annotations/GranDf_HA_GCG_train.json',
        dataset_type='grandf',
        repeats=12,
        **panorama_default_dataset_configs
    ),
    dict(
        type=PanoramaGCGDataset,
        name='GCG_Flickr30k',
        image_folder=DATA_ROOT + 'glamm_data/images/flickr30k/Flickr30K/',
        data_path=DATA_ROOT + 'glamm_data/annotations/flickr_mergedGT_GCG_train.json',
        dataset_type='flickr30k',
        repeats=0.3,
        **panorama_default_dataset_configs
    ),
    dict(
        type=PanoramaGCGDataset,
        name='GCG_OpenPsg',
        image_folder=DATA_ROOT + 'glamm_data/images/coco2017/',
        data_path=DATA_ROOT + 'glamm_data/annotations/OpenPsgGCG_train.json',
        dataset_type='openpsg',
        repeats=2,
        **panorama_default_dataset_configs
    )
]


######################### PanoCaps ####################################
# data_path is a prefix: <prefix>_caption.json and <prefix>_mask.json.
PANOCAPS_ROOT = _load_env('PANORAMA_PANOCAPS_ROOT', DATA_ROOT + 'PanoCaps/annotations/train')
PANOCAPS_IMAGES = _load_env('PANORAMA_PANOCAPS_IMAGES', DATA_ROOT + 'PanoCaps/images/train')
panorama_data_panocaps_configs = [
    dict(
        type=PanoCapsDataset,
        name='PanoCaps',
        image_folder=PANOCAPS_IMAGES,
        data_path=PANOCAPS_ROOT,
        repeats=100,
        **panorama_default_dataset_configs
    )
]


######################### COCONut-PanCap ##############################
COCONUT_ROOT = _load_env('PANORAMA_COCONUT_ROOT', DATA_ROOT + 'COCONut_PanCap/')
COCONUT_MASKS = _load_env('PANORAMA_COCONUT_MASKS', COCONUT_ROOT + 'coconut_s/panoptic')
COCO_TRAIN2017_ROOT = _load_env('PANORAMA_COCO_TRAIN2017', DATA_ROOT + 'coco/train2017/')
COCONUT_JSON = _load_env('PANORAMA_COCONUT_JSON', COCONUT_ROOT + 'coconut_s_panoptic_train2017.json')
panorama_data_coconut_configs = [
    dict(
        type=COCONutPanCapDataset,
        name='COCONutPanCap',
        image_folder=COCO_TRAIN2017_ROOT,
        data_path=COCONUT_ROOT,
        masks_dir=COCONUT_MASKS,
        panoptic_json=COCONUT_JSON,
        repeats=1,
        **panorama_default_dataset_configs
    )
]


######################### GRES ########################################
panorama_data_gres_configs = [
    dict(
        type=PanoramaGRefSeg,
        name='gRefCOCO',           # multi-target and empty-target refs
        data_root=_load_env('PANORAMA_GREFCOCO_ROOT', RES_ROOT + 'grefcoco'),
        data_prefix=dict(img_path=RES_ROOT + 'refcoco/coco2014/train2014/'),
        ann_file='instances.json',
        grefs_file='grefs(unc).json',
        split='train',
        include_multi=True,
        include_singles=False,
        include_no_target=True,
        no_target_answer='No target.',
        num_classes_per_sample=5,
        repeats=3,
        **panorama_default_dataset_configs,
    ),
    dict(
        type=PanoramaGRefSeg,
        name='gRefCOCO_singles',   # single-target refs
        data_root=_load_env('PANORAMA_GREFCOCO_ROOT', RES_ROOT + 'grefcoco'),
        data_prefix=dict(img_path=RES_ROOT + 'refcoco/coco2014/train2014/'),
        ann_file='instances.json',
        grefs_file='grefs(unc).json',
        split='train',
        include_multi=False,
        include_singles=True,
        include_no_target=False,
        num_classes_per_sample=5,
        repeats=1,
        **panorama_default_dataset_configs,
    ),
    dict(
        type=PanoramaPhraseCut,
        name='PhraseCut',
        data_root=_load_env('PANORAMA_PHRASECUT_ROOT', RES_ROOT + 'phrasecut'),
        vg_images_root=_load_env('PANORAMA_VG_IMAGES', DATA_ROOT + 'VisualGenome'),
        refer_file='VGPhraseCut_v0/refer_train.json',
        num_classes_per_sample=5,
        repeats=0.5,
        **panorama_default_dataset_configs,
    ),
]


######################### MRSeg #######################################
_mrseg_common = dict(num_classes_per_sample=1)
panorama_data_mrseg_configs = [
    dict(type=PanoramaMRSegSingleTurn, name='MRSeg_paco',       source='mr_paco',    repeats=1.0,
         **_mrseg_common, **panorama_default_dataset_configs),   # parts and instances
    dict(type=PanoramaMRSegSingleTurn, name='MRSeg_pascal',     source='mr_pascal',  repeats=1.0,
         **_mrseg_common, **panorama_default_dataset_configs),   # parts
    dict(type=PanoramaMRSegSingleTurn, name='MRSeg_attributes', source='attributes', repeats=1.0,
         **_mrseg_common, **panorama_default_dataset_configs),   # attribute clauses
    dict(type=PanoramaMRSegSingleTurn, name='MRSeg_cocostuff',  source='cocostuff',  repeats=1.0,
         **_mrseg_common, **panorama_default_dataset_configs),   # class unions
    dict(type=PanoramaMRSegSingleTurn, name='MRSeg_ade20k',     source='ade20k',     repeats=1.0,
         **_mrseg_common, **panorama_default_dataset_configs),   # class unions
    dict(type=PanoramaMRSegSingleTurn, name='MRSeg_lvis',       source='mr_lvis',    repeats=1.0,
         **_mrseg_common, **panorama_default_dataset_configs),   # instances and relations
    dict(type=PanoramaMRSegSingleTurn, name='MRSeg_vg',         source='mr_vg',      repeats=1.0,
         **_mrseg_common, **panorama_default_dataset_configs),   # relations
]

train_dataset = dict(
    type=ConcatDatasetPanorama, datasets=[
        *panorama_data_refseg_configs,
        *panorama_data_gcg_configs,
        *panorama_data_gres_configs,
        *panorama_data_panocaps_configs,
        *panorama_data_coconut_configs,
        *panorama_data_mrseg_configs,
    ]
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=panorama_collect_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = []

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed environment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
