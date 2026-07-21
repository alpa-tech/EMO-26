# SE-POSTER Reproducibility Code

This package contains the code files requested by the reviewer.

It includes:

- Explicit ResNet-18 backbone implementation
- 224 × 224 preprocessing
- RAF-DB, FERPlus, and AffectNet preparation scripts
- Training and evaluation scripts
- SE placement ablation experiments
- McNemar testing and bootstrap confidence intervals
- Grad-CAM attention visualization
- Training, comparison, confusion-matrix, and ablation plots
- Configuration files for each dataset

The package does not insert or fabricate experimental results. All reported values must be generated from the authors' real datasets, checkpoints, and prediction files.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected data layout

```text
data/
├── train/
│   ├── anger/
│   ├── disgust/
│   ├── fear/
│   ├── happiness/
│   ├── neutral/
│   ├── sadness/
│   └── surprise/
├── val/
└── test/

landmarks/
├── train/
├── val/
└── test/
```

The landmark directory must mirror the image directory.

## Prepare RAF-DB

```bash
python prepare_datasets.py rafdb \
  --image_root /path/to/RAF-DB \
  --partition_file /path/to/list_patition_label.txt \
  --output_root data/rafdb \
  --validation_ratio 0.1 \
  --seed 42
```

## Prepare FERPlus

```bash
python prepare_datasets.py ferplus \
  --image_root /path/to/FERPlus/images \
  --metadata_csv /path/to/ferplus_metadata.csv \
  --output_root data/ferplus
```

## Prepare AffectNet

```bash
python prepare_datasets.py affectnet \
  --image_root /path/to/AffectNet \
  --train_csv /path/to/training.csv \
  --test_csv /path/to/validation.csv \
  --output_root data/affectnet7 \
  --classes 7 \
  --validation_ratio 0.05 \
  --seed 42
```

## Generate facial landmark heatmaps

```bash
python prepare_landmarks.py \
  --image_root data/rafdb \
  --output_root landmarks/rafdb \
  --sigma 3
```

## Train

```bash
python train.py --config configs/rafdb.json
```

The supplied configuration explicitly records:

- ResNet-18 backbone
- 224 × 224 input size
- AdamW optimizer
- Learning rate 0.0001
- Weight decay 0.0001
- Cosine annealing
- 300 epochs
- Batch size 32
- Random seed 42
- Gradient clipping 5.0
- Embedding dimension 256
- Four Transformer encoder layers
- Eight attention heads
- SE reduction ratio 16

## Evaluate

```bash
python evaluation.py \
  --checkpoint runs/rafdb/se_all/best.pt \
  --image_dir data/rafdb/test \
  --landmark_dir landmarks/rafdb/test \
  --output_dir results/rafdb/se_all
```

The script produces `metrics.json` and `predictions.csv`.

## Statistical significance

Generate prediction files for the baseline and proposed model using the same test samples, then run:

```bash
python statistical_analysis.py \
  --baseline results/rafdb/baseline/predictions.csv \
  --proposed results/rafdb/se_all/predictions.csv \
  --output results/rafdb/statistical_analysis.json \
  --bootstrap_repetitions 10000 \
  --seed 42
```

## Ablation experiments

```bash
python run_ablations.py \
  --base_config configs/rafdb.json \
  --output_root runs/rafdb/ablations
```

The variants are:

- Baseline without SE
- Fine-level SE only
- Mid-level SE only
- Global-level SE only
- SE at all levels

## Attention visualization

```bash
python visualize_attention.py \
  --checkpoint runs/rafdb/se_all/best.pt \
  --image data/rafdb/test/happiness/test_0001.jpg \
  --landmark landmarks/rafdb/test/happiness/test_0001.png \
  --stage global \
  --output figures/attention_example.png
```

## Plot figures

```bash
python plot_figures.py training \
  --input runs/rafdb/se_all/history.csv \
  --output figures/training_curve.png
```

```bash
python plot_figures.py confusion \
  --input results/rafdb/se_all/metrics.json \
  --output figures/confusion_matrix.png
```

```bash
python plot_figures.py comparison \
  --input examples/comparison_format.csv \
  --output figures/model_comparison.png
```

```bash
python plot_figures.py ablation \
  --input runs/rafdb/ablations/ablation_results.csv \
  --output figures/ablation.png
```

## Smoke test

```bash
python smoke_test.py
```

## Important manuscript consistency

The code explicitly uses ResNet-18 and 224 × 224 inputs. The Transformer module in this implementation performs self-attention over concatenated multi-scale tokens. The manuscript should not call it cross-attention unless the architecture is changed to implement cross-attention.

The current default configuration jointly trains the landmark feature encoder. To claim that the landmark encoder is pretrained and frozen, provide the exact trained landmark checkpoint, set `freeze_landmark_encoder` to `true`, and set `landmark_checkpoint` to its path.
