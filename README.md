# Cross-domain plant disease classification: LoRA and domain-adversarial adaptation of DINOv2

Code for a leave-one-domain-out study comparing four adaptation strategies for
a DINOv2 vision foundation model on plant disease classification across three
independently collected datasets (laboratory, field, and in-the-wild web
imagery): linear probing, full fine-tuning, plain LoRA, and LoRA with a
domain-adversarial gradient-reversal head (DoGA).

Paper: Evaluating Parameter-Efficient Adaptation of a Vision Foundation Model for Plant Disease Classification Under Domain Shift (under review). 
This README documents how to reproduce Tables 1 and 3-7.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Experiments were run with PyTorch 2.7.1 (CUDA 12.6) on a single NVIDIA Tesla
P100 GPU. `requirements.txt` pins the exact package versions used.

## Data

Three datasets, filtered to 16 disease classes present in all three (see
Table 1 and Section 3.1 of the paper). Raw images are not redistributed in
this repository; the scripts below download or copy them from their original
sources.

| Dataset | Domain | Source | License |
|---|---|---|---|
| PlantVillage (Hughes and Salathe, 2015) | Laboratory | github.com/spMohanty/PlantVillage-Dataset | see source repository |
| PlantDoc (Singh et al., 2020) | Field | github.com/pratikkayal/PlantDoc-Dataset | see source repository |
| PlantWild (Wei et al., 2024) | In-the-wild, web-sourced | huggingface.co/datasets/uqtwei2/PlantWild | CC BY-NC-ND 4.0 |

```bash
# PlantVillage + PlantDoc: cloned automatically from their official
# GitHub repositories and copied into a shared folder layout.
python scripts/download_real_data.py

# PlantWild: download the "images" folder yourself from
# https://huggingface.co/datasets/uqtwei2/PlantWild (you may need to accept
# the dataset's license on the Hugging Face page first), then run:
python scripts/integrate_plantwild.py --source /path/to/plantwild/images
```

Both scripts print a per-class image count table; the totals should match
Table 1 (PlantVillage 22,099; PlantDoc 1,655; PlantWild 3,994 images across
the 16 shared classes).

PlantWild's CC BY-NC-ND 4.0 license prohibits redistributing a modified or
re-split version of its images, which is why the filtered dataset itself is
not included here; the script above lets anyone with access to the official
PlantWild release regenerate the identical derived dataset.

## Repository structure

```
configs/       Training configs (see "Reproducing the tables" below)
src/           Model, data pipeline, training loop, evaluation, statistics
scripts/       Data download/integration and multi-run experiment scripts
results/       summary.csv, rank_ablation.csv, and results/preds/ (saved
               per-image predictions used for the tables below)
```

## Reproducing the tables

**Table 3 / Table 5 (main results, generalization gap):** each of the three
leave-one-domain-out rotations was run with `scripts/run_baselines.sh`, which
trains all four methods (`doga`, `lora_plain`, `linear_probe`,
`full_finetune`) for whichever domain is set as `data.held_out_domain` in the
config it's given.

```bash
bash scripts/run_baselines.sh configs/base.yaml               # plantwild held out
bash scripts/run_baselines.sh configs/holdout_plantvillage.yaml
bash scripts/run_baselines.sh configs/holdout_plantdoc.yaml
```

Each run appends one row to `results/summary.csv` (macro-F1 on the held-out
domain, at the checkpoint selected by validation macro-F1). `scripts/
run_all_rotations.py` runs the same 12 combinations from a single base
config and reports the mean +/- sample standard deviation shown in Table 3.

**Table 4 (McNemar's test) and the bootstrap confidence intervals:** requires
per-image predictions, saved with `src/eval.py --save_dir results/preds`, one
call per method and rotation, for example:

```bash
python -m src.eval --config configs/base.yaml --mode doga \
    --checkpoint checkpoints/doga_holdout-plantwild_best.pt \
    --save_dir results/preds
python -m src.eval --config configs/base.yaml --mode lora_plain \
    --checkpoint checkpoints/lora_plain_holdout-plantwild_best.pt \
    --save_dir results/preds
```

then, once both methods' predictions exist for a rotation:

```bash
python -m scripts.compare_doga_vs_lora
```

This prints McNemar's test (with continuity correction, or the exact
binomial test when there are fewer than 25 discordant pairs) and 95%
bootstrap confidence intervals (1,000 resamples), per rotation. In Table 4,
$b_{01}$ is the count of held-out images only plain LoRA classifies
correctly, and $b_{10}$ the count only DoGA classifies correctly.

For the PlantWild rotation specifically, the predictions in
`results/preds/doga_holdout-plantwild_*` are from the $\lambda = 0.05$
(15-epoch) checkpoint, not the $\lambda = 0.3$ checkpoint reported in
Table 3, because the latter's predictions were not separately retained
before being superseded during the $\lambda$-sensitivity sweep. Both
checkpoints score below plain LoRA on this rotation (0.515 and 0.538 vs.
0.569), so this does not change the qualitative result; see the note under
Table 4 in the paper.

`scripts/compare_methods.py` is a general-purpose version of the same
comparison, for any two checkpoints and modes; its printed generalization
gap is accuracy-based, whereas Table 5 and Eq. 6 in the paper use macro-F1.

**Table 6 ($\lambda$-sensitivity sweep, PlantWild held out):** each row is one
run of `src.train` with `configs/base.yaml`'s $\lambda$ and epoch count
changed; the five configs used are included under `configs/`:

```bash
python -m src.train --config configs/lambda_0.05_ep8.yaml  --mode doga
python -m src.train --config configs/lambda_0.1_ep8.yaml   --mode doga
python -m src.train --config configs/lambda_0.5_ep8.yaml   --mode doga
python -m src.train --config configs/lambda_1.0_ep8.yaml   --mode doga
python -m src.train --config configs/lambda_0.05_ep15.yaml --mode doga
```

(`configs/base.yaml` itself, $\lambda = 0.3$ at 15 epochs, is the "0.3
(main)" row already reported in Table 3.)

**Table 7 (LoRA rank ablation, PlantWild held out):**

```bash
python scripts/rank_ablation.py --config configs/base.yaml --ranks 4 8 16 32
```

This holds `lora.alpha` fixed at its config value (16), so the LoRA scaling
$\alpha/r$ decreases as $r$ increases (4, 2, 1, 0.5 for $r$ = 4, 8, 16, 32).
Results are written to `results/rank_ablation.csv`.

## Notes

- All four adaptation methods were trained with the same optimizer (AdamW,
  learning rate $1\times10^{-4}$, weight decay $1\times10^{-4}$, constant
  schedule) and the same 15 epochs; the learning rate was not tuned
  separately per method.
- The domain classifier (used only in `doga` mode) is trained on the natural
  mix of the two training domains, without domain-balanced sampling.
-

## Citation

```bibtex
@article{devi2026evaluating,
  title   = {Evaluating Parameter-Efficient Adaptation of a Vision Foundation Model for Plant Disease Classification Under Domain Shift},
  author  = {Toijam Sonalika, Inunganbi Sanasam, Navanath Saharia},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}

```
