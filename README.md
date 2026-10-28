# Variable Star Classification with Irregularly Sampled Light Curves

Code and dataset split metadata accompanying the paper **Variable Star Classification with Irregularly Sampled Light Curves**.

The framework combines a period-aware Conv1D–BiLSTM time-series branch with a SwinV2-T image branch. The branches are trained independently, and their logits are combined by late fusion. The fusion weight is selected using the validation set.

## Files

| File | Purpose |
| --- | --- |
| `add_period_to_meta.py` | Extract period candidates and quality indicators and append period features to metadata. |
| `generate_leaves_plotimg.py` | Render time-domain light-curve images and record their paths. |
| `conv1d_bilstm_period.py` | Train and evaluate the period-aware time-series branch. |
| `plot_swinv2_tiny.py` | Train and evaluate the light-curve image branch. |
| `late_fusion_eval.py` | Load the two trained branches, select the fusion weight on the validation set, and evaluate on the test set. |
| `meta_with_plotimg.zip` | Compressed metadata containing source identifiers, class labels, fixed dataset splits, period features, and local file paths. Extract it before use. |

## Data and fixed splits

The experiments use 105,664 ASAS-SN V-band light curves selected from LEAVES (Light-curve Expandable Data Set for Automatic Classification of Variable Stars).

| Split | Number of light curves |
| --- | ---: |
| Training | 73,957 |
| Validation | 10,555 |
| Test | 21,152 |
| Total | 105,664 |

The class order used by the models is:

```text
EW, EA, SR, M, ROT, RRAB, RRC, DSCT, CEP
```

The metadata includes the following columns:

| Columns | Description |
| --- | --- |
| `id`, `label`, `band` | Source identifier, class label, and photometric band. |
| `split` | Fixed `train`, `val`, or `test` assignment. |
| `lc_path` | Path to the original light-curve CSV. |
| `png_path` | Legacy image path retained in the metadata. |
| `period_best`, `ls_power`, `pdm_theta`, `ls_power_ratio12`, `period_valid` | Estimated period, period-search quality statistics, and period-validity flag. |
| `plotimg_path`, `plotimg_exists` | Time-domain image path and image-existence flag recorded when the metadata was generated. |

Raw light curves must be obtained separately from the original LEAVES dataset. The metadata archive does not contain the raw light curves, generated images, or trained model checkpoints. Preserve the supplied identifiers, labels, and split assignments when reproducing the experiments.

## Dependencies

The scripts use Python and the following packages:

```text
torch
torchvision
numpy
pandas
scipy
matplotlib
Pillow
tqdm
```

Install compatible PyTorch and torchvision builds for your hardware. A CUDA-capable GPU is recommended for training. Image-branch training uses ImageNet-pretrained SwinV2-T weights; these weights must be downloaded or cached locally before training in an offline environment.

## Configure local paths

These are research scripts configured through constants near the beginning of each file, rather than command-line arguments.

1. Extract `meta_with_plotimg.zip` and locate the metadata CSV.
2. Obtain the raw LEAVES light curves.
3. Update metadata inputs and outputs (`CSV_PATH`, `META_IN`, and `META_OUT`), data roots (`LC_ROOT`, `PNG_ROOT`, `IMG_ROOT`, and `OUT_ROOT`), and checkpoint paths as applicable.
4. The distributed metadata retains absolute paths from the original computing environment. Replace or remap these paths to your local files. Changing a root-directory setting alone does not override an absolute path stored in the CSV. Keep identifiers, labels, splits, and numerical features unchanged.
5. Ensure that each script reads the intended metadata file. Filenames containing `meta_with_period`, `meta_with_plotimg`, or `_320` must be configured consistently with your local files.

The image-existence flag describes the original environment; it does not verify that files exist on a new machine. Make sure all validation and test samples are available before evaluation.

## Workflow

The supplied metadata already contains period features and fixed split assignments. Recomputing period features is optional:

```bash
python add_period_to_meta.py
```

Generate time-domain light-curve images after configuring the metadata input and image output directory. For the paper's image configuration, set the rendering size to 320 × 320 pixels and use a model input size of 320:

```bash
python generate_leaves_plotimg.py
```

Train the two branches independently:

```bash
python conv1d_bilstm_period.py
python plot_swinv2_tiny.py
```

Configure `TIME_CKPT` and `IMG_CKPT` to point to the corresponding trained checkpoints, then run:

```bash
python late_fusion_eval.py
```

Keep the class order and model architecture consistent between training and fusion evaluation. Use the same fixed splits throughout. The validation set is used to select the fusion weight; the test set is used for final evaluation. Results from a new training run can vary with the random seed, software environment, and hardware.

## Contact

Corresponding author: Xiao-Jing Liu, Qinghai University.  
Email: liuxj@qhu.edu.cn
