# Model files

The apps (web + mobile) load models from the **portable DigitLab JSON format**
produced by the Kaggle notebooks (see `kaggle/README.md`).

## Contents

All files below come from the full training runs on Kaggle (dataset
`hojjatk/mnist-dataset`, 60k train / 10k test), downloaded from the kernels'
output folders:

| file | model | test accuracy | kernel |
|---|---|---|---|
| `svc_mnist.json` | RBF-SVM (OvR export, 11k support vectors) | **97.65%** (canonical SVC 97.57%) | `digitlab-01-classical-ml-rbf-svm` |
| `mlp_mnist.json` | MLP 784→512→256→10 (BN + dropout 0.2) | **99.07%** | `digitlab-02-neural-network-mlp` |
| `cnn_mnist.json` | CNN width-48, 2 conv blocks (750k params) | **99.57%** | `digitlab-03-deep-learning-cnn` |

The framework-format counterparts are committed alongside: `svc_mnist.joblib`
(scikit-learn), `mlp_mnist.pt` / `cnn_mnist.pt` (PyTorch state dicts) — load
them with `joblib.load(...)` / `torch.load(...)`.

`metrics_*.json` carry the full hyperparameter-search tables, final metrics,
training curves and artifact hashes. Each notebook also asserts export parity
(JSON recompute ≡ framework predictions — 100% agreement for all three).

## Swapping in a re-run

Download the `models/` output folder from each Kaggle notebook and replace the
files here, then:

```bash
cd web    && npm run sync && npm run build     # refreshes web/public/models
cd mobile && npm run sync                       # refreshes mobile/assets/models
```

and redeploy (`bash tools/deploy-pages.sh`) / rebuild the APK.
