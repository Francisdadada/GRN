# Training

Training is config-driven:

```bash
python scripts/train.py --config configs/train_0.05.yaml
```

The training loop includes:

- supervised Dice loss on labeled images,
- adversarial generator/discriminator updates on unlabeled images,
- pixel reconstruction loss to keep generated images close to inputs,
- ICT-style consistency loss on mixed unlabeled images,
- validation loss checkpointing.

Outputs are saved under `artifacts/checkpoints/...` by default.

## Data Layout

Expected dataset layout:

```text
dataset/
  0.05/
    labelled_img/
    labelled_msk/
    unlabelled_img/
  val_img/
  val_msk/
```

The config also supports the American spelling `labeled_*` and `unlabeled_*`.
