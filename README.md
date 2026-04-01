<div align="center">

# 【CVPR 2026 MaCVi】 A unified Benchmark for Multi-Frame Image Restoration under Severe Refractive Warping
</div>

## [Paper]()

## [Dataset] () 

## Performance Evaluation

| argument | description |
|---|---|
| --dataset_root | root directory of the above dataset with backgrounds and wave profiles. |
| --wave_type | wave type ocean/shallow/sine/ripple |
| --amplitude | wave amplitude low/mid/high/extreme |
| --L | number of frames in eval |
| --method | evaluated method ref/mean/grid_registration/datum |

python eval.py --L 1 --wtype ocean --amplitude low --method ref
python eval.py --L 49 --wtype ripple --amplitude mid --method datum


## Citation
Please consider citing our work as follows if it is helpful.
```

```
