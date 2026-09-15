# PANORAMA-4B finetuned on RefCOCO/+/g: starts from the panorama_4b checkpoint and trains on the
# referring-segmentation data only; everything else is inherited.
from mmengine.config import read_base

with read_base():
    from .panorama_4b import *  # noqa: F401,F403

# Init from the parent run's last checkpoint (<PANORAMA_RESULTS_ROOT>/panorama_4b/train/last_checkpoint)
# or from PANORAMA_FT_INIT_PTH in .env.
_ft_parent_run = 'panorama_4b'
pretrained_pth = _load_env('PANORAMA_FT_INIT_PTH', '')  # noqa: F405
if not pretrained_pth:
    _results_root = _load_env('PANORAMA_RESULTS_ROOT', '')  # noqa: F405
    assert _results_root, 'set PANORAMA_RESULTS_ROOT in .env'
    _last_ckpt_file = _results_root + '/' + _ft_parent_run + '/train/last_checkpoint'
    try:
        with open(_last_ckpt_file) as _f:
            pretrained_pth = _f.read().strip()
        del _f          # config-level names are pickled
    except OSError:
        pretrained_pth = ''
    assert pretrained_pth, (
        'no init weights: ' + _last_ckpt_file + ' is missing or empty. Run panorama_4b '
        'first (or set PANORAMA_FT_INIT_PTH in .env to a specific .pth).')
# Mutate the inherited dicts; reassigning would drop the base keys.
model['pretrained_pth'] = pretrained_pth  # noqa: F405

# Referring-segmentation data at the mixture repeats; lower _FT_SCALE to shrink it uniformly.
_FT_SCALE = 1.0
_ft_sources = [
    _d for _d in panorama_data_refseg_configs  # noqa: F405

]
for _d in _ft_sources:
    _d['repeats'] = _d['repeats'] * _FT_SCALE

train_dataset = dict(
    type=ConcatDatasetPanorama,  # noqa: F405
    datasets=[*_ft_sources],
)
# Swap only the dataset, keep batch_size/sampler/collate_fn/num_workers from the base.
train_dataloader['dataset'] = train_dataset  # noqa: F405
