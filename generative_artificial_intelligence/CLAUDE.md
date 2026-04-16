# Generative Artificial Intelligence - KU Homework

## Overview
Korea University coursework on generative models. All assignments are PyTorch-based implementations of core generative modeling techniques.

## Project Structure
```
generative_artificial_intelligence/
├── assignment_1/          # Normalizing Flows
│   ├── Q1/                # 1D Flow (Mixture of Gaussians CDF/PDF)
│   │   ├── model.py       # Flow1d model definition
│   │   ├── main.py        # Train/eval loop (NLL loss with Uniform target)
│   │   ├── data.py        # Data loader
│   │   └── Q1.ipynb       # Execution notebook
│   └── Q2/                # Composable Flows (Flow + LogitTransform chain)
│       ├── model.py       # Flow1d, LogitTransform, FlowComposable1d
│       ├── main.py        # Train/eval loop (log-space dz/dx, Normal target)
│       ├── data.py        # Data loader
│       └── ComposableFlows1d.ipynb
├── assignment_2/          # Autoregressive Models (Histogram, PixelRNN)
│   ├── main.ipynb         # Q1: Histogram fitting via softmax MLE
│   │                      # Q2a: PixelRNN (GRU-based, binary image generation)
│   │                      # Q2b: Inference - NLL comparison with bit-flip perturbation
│   ├── hw1_helper.py      # Visualization and result-saving helpers
│   ├── pytorch_util.py
│   ├── utils.py
│   └── data/              # mnist.pkl, shapes.pkl
└── assignment_3/          # Variational Autoencoders
    ├── variational_autoencoders.ipynb  # Main notebook (VAE + CVAE on MNIST)
    ├── vae.py             # VAE, CVAE, reparametrize(), loss_function()
    ├── helper.py          # train_vae(), show_images(), one_hot()
    └── utils/             # grad, utils utilities
```

## Tech Stack
- Python 3.12, PyTorch (CUDA)
- Key libraries: torchvision, matplotlib, numpy

## Key Patterns
- Implementation locations are marked with `### FILL IN ###` or `# TODO` comments
- Jupyter notebooks are the primary execution entry points; results saved to `results/` or `save/`
- Each assignment is independent with its own data and models

## Commands
```bash
# Run Jupyter notebook (inside container)
jupyter notebook --ip=0.0.0.0 --port=8888

# Run specific assignment
cd assignment_1/Q1 && python main.py
```

## Notes
- assignment_2/main.ipynb is based on Berkeley deepul hw1, Colab-compatible
- In assignment_3 vae.py, loss_function arg order is (x_hat, x, mu, logvar), but helper.py calls it as (recon_batch, data, mu, logvar)
- GPU (CUDA) required for assignment_3 (helper.py has hardcoded `device='cuda:0'`)
