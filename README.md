# L2 English Speech Scoring (Example Code Release)

This repository provides an **example-oriented** code release for a multimodal + multitask L2 English speech scoring pipeline.

It is shared for portfolio review and implementation reference (not a full final reproduction package).

## What This Repository Shows
- Multimodal architecture design with speech/text feature integration
- Multitask prediction pipeline for multiple scoring dimensions
- Core training/evaluation code organization
- Example notebooks with outputs removed

## Project Structure
```bash
.
├── src/                    # training/evaluation pipeline
│   ├── train.py
│   ├── models.py
│   ├── models_with_trait_attention.py
│   ├── dataset.py
│   ├── eval_metrics.py
│   ├── ctc.py
│   └── utils.py
├── modules/                # transformer/attention components
│   ├── transformer.py
│   ├── multihead_attention.py
│   └── position_embedding.py
└── notebooks/              # sanitized example notebooks (no result outputs)
```

## Notes
- Experimental outputs are intentionally removed.
- Baseline-comparison notebooks are intentionally excluded in this release.
- Dataset and large external resources are not bundled.
