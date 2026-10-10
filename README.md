# DigitLab — MNIST Digit Recognition: Three Models, Web & Mobile

Train three models on MNIST — a classical **RBF-SVM**, a fully-connected **MLP**,
and a small **CNN** — then recognize **your own hand-drawn digits** with any of
them, in the browser and on Android, using a dependency-free TypeScript
inference core. No server, no ML runtime: the models ship inside the apps.

**Live demo:** <https://aliomarabdelhady.github.io/digitlab/>
**Android APK:** <https://aliomarabdelhady.github.io/digitlab/digitlab.apk>

## Results (official 10k-image MNIST test set)

| model | test accuracy | macro F1 | size / notes |
|---|---|---|---|
| LogisticRegression (baseline) | 91.690% | — | reference |
| RandomForest-300 (baseline) | 96.020% | — | reference |
| **SVC (RBF kernel)** | **97.570%** (canonical) / 97.650% (OvR export) | 97.57% | 11,107 support vectors |
| **MLP (784→512→256→10)** | **99.070%** | 99.061% | 537k params · early stop @ ep 26/34 |
| **CNN (2 conv blocks, w=48)** | **99.570%** | 99.568% | 750k params · early stop @ ep 19/24 |

**Selected model: the CNN** — lowest error (0.43%), strongest robustness to
handwriting variation, still a small lecture-style network that runs in
milliseconds on-device. Full reasoning in [`REPORT.md`](REPORT.md).

## Repository layout

| path | contents |
|---|---|
| `kaggle/` | the three training notebooks (`.ipynb`) + run instructions |
| `models/` | trained models — portable `*_mnist.json` (apps), `*.pt` / `*.joblib` (framework formats), `metrics_*.json` (search tables, curves, hashes) |
| `core/` | shared TypeScript inference core: drawing→784 MNIST preprocessing + SVC/MLP/CNN engines + parity tests |
| `web/` | web app (Vite + React): drawing canvas, model switcher, confidence UI |
| `mobile/` | Android app (Expo / React Native) with the same design |
| `figures/` | learning curves + confusion matrices from the training runs |
| `tools/` | fixture generator, Kaggle output fetcher, Pages deploy script |

## How it fits together

```
Kaggle notebooks ──▶ models/*.joblib|*.pt   (framework formats)
                    models/*.json          (portable: plain weight arrays)
                                │
      ┌─────────────────────────┴────────────────────────┐
 web/ (browser)                                      mobile/ (bundled in APK)
      └───────── both run core/ — pure-TS engines ──────┘
```

All three models use one preprocessing pipeline, at train time and at serve
time: pixels scaled to `[0,1]` (÷255), nothing else. The apps reproduce how
MNIST itself was built — crop the ink bounding box, scale the longer side to
20 px, center by center of mass in a 28×28 field — which is why canvas and
finger drawings classify correctly.

## Running everything

### Train (Kaggle)
See [`kaggle/README.md`](kaggle/README.md): import each notebook, attach the
dataset [`hojjatk/mnist-dataset`](https://www.kaggle.com/datasets/hojjatk/mnist-dataset),
run. Outputs land in `models/`.

### Web
```bash
cd web
npm install
npm run sync        # copy models/*.json → public/models/
npm run dev         # http://localhost:5173
npm run build       # static production build (relative paths)
```

### Mobile (Android APK)
```bash
cd mobile
npm install
npm run sync        # copy models/*.json → assets/models/*.dlmodel
npx expo prebuild -p android
cd android && ./gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a
adb install -r app/build/outputs/apk/release/app-release.apk
```
Requires Android SDK + JDK 17. The release APK is debug-signed (fine for
personal sideloading; use your own keystore for store distribution).

### Tests
```bash
cd core && npm install && npm test     # engine parity vs recorded fixtures
```

## Verification highlights

- Every notebook ends with an **export-parity check**: predictions recomputed
  from the portable JSON (with the exact arithmetic the app's engine performs)
  must match the framework's own predictions — 100% agreement on all three.
- `core/` tests re-verify the shipped models against recorded predictions on
  256 fixed MNIST test images.
- Training curves show best-epoch restores with ≤0.3-pt train/validation gap
  (MLP) and a negative gap (CNN) — no under- or overfitting. Figures in
  `figures/` and inside the notebooks.

## License

MIT — see [LICENSE](LICENSE).
