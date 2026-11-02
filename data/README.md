# Data & Weights

The data goes under `$DATA_ROOT` in this layout:

```
$DATA_ROOT/
├── seaclips/      sea-cocovid-{train,val,test}.json + images
├── lars/          lars_v1.0.0_images/ + lars_{train,val}_coco.json + lars_test_empty.json
├── modd2/         modd2_{train,val}_coco.json + images
├── mods/          mods_obstacle_test_coco.json + images (held-out test only)
└── pretrained/    dinov3_vitb16_pretrain_lvd1689.pth
```
> [!NOTE]
> Consider that the datasets and weights are released under different licenses than this code repository. You must comply with them when using the corresponding datasets and weights.


## Datasets
| Dataset | Source | Preparation |
|---|---|---|
| SeaClips | [link](https://huggingface.co/datasets/SEA-AI/SeaClips) | No need for convert script |
| LaRS | [link](https://lojzezust.github.io/lars-dataset/) | `python data/convert_lars.py` writes the COCO JSONs; for test export (`scripts/export_lars_test.py`), copy [`example/lars_test_empty.json`](https://github.com/JonNatanael/macvi-usv-odce-toolkit-2024/blob/main/example/lars_test_empty.json) from the [MaCVi toolkit](https://github.com/JonNatanael/macvi-usv-odce-toolkit-2024) to `$DATA_ROOT/lars/lars_test_empty.json` |
| MODD2 | [link](https://box.vicos.si/borja/viamaro/index.html) | `python data/convert_modd2.py` writes the COCO JSONs and copies the frames into `train/`, `val/` |
| MODS | [link](https://box.vicos.si/borja/viamaro/index.html) | `python data/convert_mods.py` writes the COCO JSON (all obstacle types as one class) |

## DINOv3 weights

The backbone is the DINOv3 ViT-B/16 pretrained on LVD-1689M. Request access
and download the weights on Meta's
[DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
The configs expect the checkpoint at `$DATA_ROOT/pretrained/dinov3_vitb16_pretrain_lvd1689.pth`. For this, rename
the downloaded file or explicitly set `--backbone.dinov3_weights_path`.
