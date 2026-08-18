# MAVD: Enhancing Low-Resource Software Vulnerability Detection through Data Augmentation and Multi-Task Learning

Replication package: source code and the processed datasets used for the main results.

## Requirements

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | 2.1.0 (CUDA 12.1, cuDNN 8) |
| transformers | 4.36.2 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| numpy | 1.26.0 |
| scipy | 1.15.3 |

`transformers` must match the torch version. Version 4.36.2 works with torch 2.1.0; transformers 4.57 and 5.x require torch>=2.2.

Install torch from the CUDA build that matches the machine, then the rest:

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The backbone `microsoft/codebert-base` (about 500 MB) is downloaded from HuggingFace on the first run and cached afterwards.

## Layout

```
MAVD/
  requirements.txt
  src/
    train_v2_multitask.py     MAVD trainer
    train_baseline.py         single-head CodeBERT baseline
    mavd.py                   model definitions
    dann_utils.py             language classifiers
    ultis.py                  data loading
    seed_utils.py             seeding
    logger.py                 logging
    dump_probs.py             writes per-sample probabilities
    augment/
      augment_python.py       label-preserving Python augmentation
    comparative/
      moevd.py                MoEVD expert and router
      moevd_pipeline.py       MoEVD training and inference
      train_splvd.py          SPLVD self-paced training
      eval_comparative.py     metric aggregation
  dataset/
    sven_python/              target language, 760 functions, 5 folds
    primevul_ccpp/            C/C++ support, 220 functions
    cleanvul_js/              JavaScript support, 552 functions
    augmented/                augmented target training data, 5 folds
```

Source modules import each other by module name, so `src` must be on the import path.

## Datasets

All source functions are comment-free. Each line is one JSON record.

| Directory | File | Functions | Role |
|---|---|---|---|
| `dataset/sven_python/` | `data.jsonl` | 760 | target language |
| `dataset/sven_python/fold{1..5}/` | `train.jsonl`, `val.jsonl`, `test.jsonl` | 456 / 152 / 152 | fixed 5-fold split |
| `dataset/primevul_ccpp/` | `train_vul.jsonl`, `train_non_vul.jsonl` | 110 + 110 | C/C++ support |
| `dataset/cleanvul_js/` | `four_cwe_js_cleanvul_no_compress_norm.jsonl` | 552 | JavaScript support |
| `dataset/augmented/fold{1..5}/` | `train_aug.jsonl` | 2,146 to 2,168 | augmented target training data |

The five test folds are frozen and identical across every configuration, so each target function is used as a test sample exactly once per seed. Augmentation is applied to the training split only, so no augmented variant of a test function is ever seen during training.

## Regenerating the augmented data

`dataset/augmented/` is shipped because it is the exact input behind the reported numbers, so training can start at Step 1 without regenerating anything.

The generator is included for inspection and can be rerun into a separate directory:

```bash
export PYTHONPATH=src
for F in 1 2 3 4 5; do
  python src/augment/augment_python.py \
    --in dataset/sven_python/fold$F/train.jsonl \
    --out regenerated/fold$F/train_aug.jsonl \
    --k 1 --enhanced --per_transform --seed 42
done
```

The output is not byte-identical to `dataset/augmented/` on a different interpreter. The transformations parse each function with `ast` and re-emit it with `ast.unparse`, and both stages changed behaviour across Python releases. Regenerating under Python 3.12 reproduces about 82 percent of the shipped fold-1 records and yields 2,151 instead of 2,154. Use `dataset/augmented/` to reproduce the reported results, and treat regeneration as a way to obtain new augmented data rather than the same data.

## Step 1: train MAVD

Full configuration, one seed and one fold. Substitute `SEED` and `F` to cover the grid.

```bash
export PYTHONPATH=src
SEED=42
F=1
OUT=results/mavd/seed$SEED/fold$F

python src/train_v2_multitask.py \
  --train_files dataset/sven_python/fold$F/train.jsonl \
                dataset/augmented/fold$F/train_aug.jsonl \
                dataset/primevul_ccpp/train_vul.jsonl \
                dataset/primevul_ccpp/train_non_vul.jsonl \
                dataset/cleanvul_js/four_cwe_js_cleanvul_no_compress_norm.jsonl \
  --val_files dataset/sven_python/fold$F/val.jsonl \
  --epochs 20 --batch_size 12 --learning_rate 2e-5 \
  --uncertainty --ccpp_head --js_head \
  --lang_adapter --adapter_dim 64 --adapter_lr 1e-4 \
  --ccpp_loss_scale 1.0 --js_loss_scale 1.0 \
  --ccpp_start_epoch 3 --js_start_epoch 3 \
  --pcgrad --lr_warmup_ratio 0.1 \
  --allow_unmapped_group --seed $SEED \
  --select_metric macro --output_dir $OUT/model
```

## Step 2: write probabilities

Metrics are computed from these files, not from the training log. Use `--model_type adapter` whenever `--lang_adapter` was set during training, `v2` otherwise, and `baseline` for the single-head model.

```bash
python src/dump_probs.py --model_type adapter --num_groups 2 \
  --ckpt $OUT/model/v2_multitask_best.pth \
  --files dataset/sven_python/fold$F/test.jsonl --out_csv $OUT/test_probs.csv

python src/dump_probs.py --model_type adapter --num_groups 2 \
  --ckpt $OUT/model/v2_multitask_best.pth \
  --files dataset/sven_python/fold$F/val.jsonl --out_csv $OUT/val_probs.csv
```

## Example configurations

Every configuration below keeps the arguments of Step 1 and replaces only the flag block. All use the same folds, the same decision threshold of 0.5, and the same CodeBERT encoder.

| Configuration | Flag block |
|---|---|
| Multi-task, target only | `--uncertainty --allow_unmapped_group` |
| `+ C/C++` | `--uncertainty --ccpp_head --lang_adapter --adapter_dim 64 --adapter_lr 1e-4 --ccpp_loss_scale 1.0 --ccpp_start_epoch 3 --pcgrad --lr_warmup_ratio 0.1` |
| `+ JS` | `--uncertainty --js_head --lang_adapter --adapter_dim 64 --adapter_lr 1e-4 --js_loss_scale 1.0 --js_start_epoch 3 --pcgrad --lr_warmup_ratio 0.1` |
| MAVD (full) | flag block of Step 1 |
| Adapter only | full block with `--ccpp_start_epoch 0 --js_start_epoch 0` and `--pcgrad` removed |
| Delay only | full block with `--pcgrad` removed |
| PCGrad only | full block with `--ccpp_start_epoch 0 --js_start_epoch 0` |

Add or drop `dataset/augmented/fold$F/train_aug.jsonl` from `--train_files` to switch a configuration between augmented and raw data.

## Baselines

Single-head CodeBERT:

```bash
python src/train_baseline.py \
  --train_files dataset/sven_python/fold$F/train.jsonl \
  --val_files dataset/sven_python/fold$F/val.jsonl \
  --epochs 20 --batch_size 12 --learning_rate 2e-5 \
  --seed $SEED --select_metric macro --output_dir results/baseline/seed$SEED/fold$F/model
```

MoEVD, up to 25 epochs for the experts at batch size 12 followed by up to 15 epochs for the router at batch size 16:

```bash
export PYTHONPATH=src:src/comparative
python src/comparative/moevd_pipeline.py \
  --data_dir dataset/sven_python --fold $F --seed $SEED --mode group2 \
  --extra_train_files dataset/augmented/fold$F/train_aug.jsonl \
  --out_dir results/moevd_group2/seed$SEED/fold$F
```

SPLVD, up to 40 epochs at batch size 8 with an early-stopping patience of 10, keeping the self-paced scheduler parameters of the original design:

```bash
export PYTHONPATH=src:src/comparative
python src/comparative/train_splvd.py \
  --train_file dataset/sven_python/fold$F/train.jsonl \
  --val_file dataset/sven_python/fold$F/val.jsonl \
  --test_file dataset/sven_python/fold$F/test.jsonl \
  --seed $SEED --batch_size 8 \
  --extra_train_files dataset/augmented/fold$F/train_aug.jsonl \
  --out_dir results/splvd/seed$SEED/fold$F
```

Drop `--extra_train_files` to obtain the raw rows of either baseline.

## Reproducing the reported protocol

Each configuration is evaluated over five folds under three seeds, which gives 15 matched runs.

```bash
for SEED in 42 7 1234; do
  for F in 1 2 3 4 5; do
    :  # Step 1 followed by Step 2
  done
done
```

Two folds fit in 16 GB at batch size 12. Every run writes `model/`, `test_probs.csv` and `val_probs.csv` under its own `seed$SEED/fold$F` directory.

Aggregation reads `<root>/<configuration>/seed<S>/fold<F>/test_probs.csv`, so the `--output_dir` of each run determines the configuration name it is aggregated under.

```bash
export PYTHONPATH=src:src/comparative
python src/comparative/eval_comparative.py \
  --root results --seeds 42,7,1234 --folds 1,2,3,4,5
```

`CONFIGS` at the top of `eval_comparative.py` lists the configuration directories to report and the label used for each. It already contains `baseline`, `moevd_group2` and `splvd`. Add an entry such as `("MAVD", "mavd")` for any directory name not yet listed, otherwise that configuration is skipped.
