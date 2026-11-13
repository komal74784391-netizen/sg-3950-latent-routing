# Latent Routing Networks

Official implementation for the paper:

> **Latent Routing Networks for Efficient Sequence Modeling**
> Stephen Gomez, Priya Natarajan, Daniel Okafor
> Transactions on Machine Learning Research (2024)

The complete implementation is available under `src/latent_routing/`. The main
experiment configuration lives in `configs/base.yaml`.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m latent_routing.train --config configs/base.yaml
```

## Citation

```bibtex
@article{gomez2024latent,
  title   = {Latent Routing Networks for Efficient Sequence Modeling},
  author  = {Gomez, Stephen and Natarajan, Priya and Okafor, Daniel},
  journal = {Transactions on Machine Learning Research},
  year    = {2024}
}
```
