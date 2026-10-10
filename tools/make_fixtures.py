#!/usr/bin/env python3
"""Regenerate core/test/fixtures/fixtures.json — the recorded predictions the
TypeScript engine tests are checked against.

Reads the installed models/*.json and a local copy of the MNIST IDX test files
(the canonical four files, e.g. from the Kaggle dataset), runs each model with
the same arithmetic the JS engine performs, and stores 256 images + expected
probabilities.

Usage:  python tools/make_fixtures.py --idx-dir /path/to/mnist_idx_files
Requires: numpy  (pip install -r tools/requirements.txt)
"""
import argparse
import base64
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
N = 256


def load_idx_images(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n, h, w = np.frombuffer(f.read(16), dtype=">u4")
        assert magic == 2051, f"bad magic in {path}"
        return np.frombuffer(f.read(n * h * w), dtype=np.uint8).reshape(n, 784)


def load_idx_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n = np.frombuffer(f.read(8), dtype=">u4")
        assert magic == 2049, f"bad magic in {path}"
        return np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def svc_forward(blob, X01):
    m = blob["model"]
    sv = np.frombuffer(base64.b64decode(m["sv_u8"]), np.uint8).astype(np.float32).reshape(-1, 784) / 255.0
    dual = np.frombuffer(base64.b64decode(m["dual_f32"]), "<f4").reshape(-1, 10)
    b = np.frombuffer(base64.b64decode(m["intercept_f32"]), "<f4")
    g = m["gamma"]
    scores = np.empty((len(X01), 10), np.float32)
    sv2 = (sv ** 2).sum(1)[None, :]
    for i0 in range(0, len(X01), 256):
        xb = X01[i0:i0 + 256]
        d2 = np.maximum((xb ** 2).sum(1, keepdims=True) + sv2 - 2.0 * (xb @ sv.T), 0.0)
        scores[i0:i0 + 256] = np.exp(-g * d2) @ dual + b[None, :]
    return softmax_np(scores)


def mlp_forward(blob, X01):
    x = X01.astype(np.float32)
    layers = blob["model"]["layers"]
    for i, L in enumerate(layers):
        w = np.frombuffer(base64.b64decode(L["w"]), "<f4").reshape(L["out"], L["in"])
        b = np.frombuffer(base64.b64decode(L["b"]), "<f4")
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0.0)
    return softmax_np(x)


def cnn_forward(blob, X01):
    from numpy.lib.stride_tricks import sliding_window_view
    x = X01.reshape(-1, 1, 28, 28).astype(np.float32)
    for op in blob["model"]["ops"]:
        kind = op["op"]
        if kind == "conv":
            w = np.frombuffer(base64.b64decode(op["w"]), "<f4").reshape(op["out_c"], op["in_c"], op["k"], op["k"])
            b = np.frombuffer(base64.b64decode(op["b"]), "<f4")
            xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
            win = sliding_window_view(xp, (3, 3), axis=(2, 3))
            x = np.einsum("nchwij,ocij->nohw", win, w, optimize=True) + b[None, :, None, None]
        elif kind == "linear":
            w = np.frombuffer(base64.b64decode(op["w"]), "<f4").reshape(op["out"], op["in"])
            b = np.frombuffer(base64.b64decode(op["b"]), "<f4")
            x = x.reshape(x.shape[0], -1) @ w.T + b
        elif kind == "relu":
            x = np.maximum(x, 0.0)
        elif kind == "maxpool2":
            n, c, h, w_ = x.shape
            x = x.reshape(n, c, h // 2, 2, w_ // 2, 2).max(axis=(3, 5))
        elif kind == "flatten":
            x = x.reshape(x.shape[0], -1)
        elif kind == "softmax":
            return softmax_np(x)
        else:
            raise ValueError(kind)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx-dir", required=True,
                    help="folder with t10k-images.idx3-ubyte / t10k-labels.idx1-ubyte")
    ap.add_argument("--out", default=str(ROOT / "core" / "test" / "fixtures" / "fixtures.json"))
    args = ap.parse_args()

    d = args.idx_dir
    img_path = next(p for p in [
        pathlib.Path(d) / "t10k-images-idx3-ubyte", pathlib.Path(d) / "t10k-images.idx3-ubyte"]
        if p.exists())
    lbl_path = next(p for p in [
        pathlib.Path(d) / "t10k-labels-idx1-ubyte", pathlib.Path(d) / "t10k-labels.idx1-ubyte"]
        if p.exists())
    X, y = load_idx_images(str(img_path)), load_idx_labels(str(lbl_path))

    rng = np.random.default_rng(123)
    idx = np.concatenate([
        rng.choice(np.where(y == c)[0], N // 10 + (1 if c < N % 10 else 0), replace=False)
        for c in range(10)
    ])
    imgs = X[idx]
    X01 = imgs.astype(np.float32) / 255.0

    fixtures = {
        "n": int(N),
        "labels": y[idx].tolist(),
        "images_u8": base64.b64encode(imgs.tobytes()).decode("ascii"),
        "models": {},
    }
    for kind, fwd in [("svc", svc_forward), ("mlp", mlp_forward), ("cnn", cnn_forward)]:
        path = ROOT / "models" / f"{kind}_mnist.json"
        blob = json.loads(path.read_text())
        probs = fwd(blob, X01).astype(np.float64)
        fixtures["models"][kind] = {
            "preds": probs.argmax(1).tolist(),
            "probs": [[round(float(p), 6) for p in row] for row in probs],
        }
        acc = float((probs.argmax(1) == y[idx]).mean())
        print(f"{kind}: sampled-set accuracy {acc*100:.2f}%")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixtures))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
