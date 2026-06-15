# VIL-DualQueue-SFCM

This repository provides the model and training implementation of a VIL radar-echo forecasting framework.

## Included

- encoder, spatiotemporal predictor, and decoder network layers;
- observation-queue channel attention and future-queue temporal attention;
- adaptive weak-echo masking;
- native-filled split dual-queue construction and stepwise filling;
- Spatiotemporal Fusion Calibration Module;
- Dual-Supervised Echo-Structure Constraint Loss;
- complete model forward propagation;
- complete training loop and essential experimental configuration.

## Tensor Contract

```text
Input:  [B, 10, 1, H, W]
Target: [B, 12, 1, H, W]
Range:  [0, 1]
```

## Experimental Configuration

- Random seed: `42`
- Dataset split: `80% / 10% / 10%`
- Maximum epochs: `150`
- Optimizer: AdamW

Additional model and loss parameters are listed in `config.py`.

## Training

The `train` function accepts a prepared PyTorch DataLoader:

```python
from config import create_parser
from train import build_training_interfaces, train

args = create_parser().parse_args([])
model, criterion = build_training_interfaces(args)
model = train(model, criterion, train_loader, args)
```

The private dataset acquisition and preparation pipeline, dataset files, trained weights, and reported experimental outputs are not included.

## Files

```text
config.py
data_pipeline.py
modules.py
model.py
loss_functions.py
train.py
```

## License

See `LICENSE`.
