"""Train a small MLP on the scikit-learn digits dataset with plain NumPy, then
export fp32 and int8 .tinf models plus a held-out test set for the C runtime.

    python tools/train_export.py --out models

Outputs
    models/digits_f32.tinf    fp32 weights
    models/digits_i8.tinf     int8 weights, per-output-channel symmetric scales
    models/digits_test.bin    u32 n | u32 dim | f32 X[n*dim] | u32 y[n] | f32 ref_logits[n*10]
"""
import argparse
import os
import struct

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

DENSE, RELU, SOFTMAX = 1, 2, 3
F32, I8 = 0, 1


def train(X, y, sizes, epochs=60, lr=0.05, batch=64, seed=0):
    rng = np.random.default_rng(seed)
    Ws, bs = [], []
    for a, b in zip(sizes[:-1], sizes[1:]):
        Ws.append(rng.normal(0, np.sqrt(2.0 / a), (b, a)).astype(np.float32))  # He init, [out][in]
        bs.append(np.zeros(b, np.float32))
    Y = np.eye(sizes[-1], dtype=np.float32)[y]
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            xb, yb = X[idx[s:s + batch]], Y[idx[s:s + batch]]
            acts = [xb]
            for k, (W, b) in enumerate(zip(Ws, bs)):
                z = acts[-1] @ W.T + b
                acts.append(np.maximum(z, 0) if k < len(Ws) - 1 else z)
            z = acts[-1] - acts[-1].max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            g = (p - yb) / len(xb)
            for k in range(len(Ws) - 1, -1, -1):
                gW, gb = g.T @ acts[k], g.sum(0)
                if k:
                    g = (g @ Ws[k]) * (acts[k] > 0)
                Ws[k] -= lr * gW
                bs[k] -= lr * gb
        lr *= 0.97
    return Ws, bs


def forward(X, Ws, bs):
    h = X
    for k, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W.T + b
        if k < len(Ws) - 1:
            h = np.maximum(h, 0)
    return h


def write_model(path, Ws, bs, dtype):
    layers = []
    for k, (W, b) in enumerate(zip(Ws, bs)):
        layers.append((DENSE, W, b))
        if k < len(Ws) - 1:
            layers.append((RELU, W.shape[0]))
    layers.append((SOFTMAX, Ws[-1].shape[0]))
    with open(path, "wb") as f:
        f.write(b"TINF" + struct.pack("<II", 1, len(layers)))
        for L in layers:
            if L[0] == DENSE:
                _, W, b = L
                out, inn = W.shape
                f.write(struct.pack("<IIII", DENSE, inn, out, dtype))
                if dtype == F32:
                    f.write(W.astype("<f4").tobytes())
                else:
                    scale = np.abs(W).max(1) / 127.0
                    scale[scale == 0] = 1.0
                    q = np.clip(np.rint(W / scale[:, None]), -127, 127).astype(np.int8)
                    f.write(scale.astype("<f4").tobytes())
                    f.write(q.tobytes())
                    f.write(b"\0" * ((4 - q.size % 4) % 4))
                f.write(b.astype("<f4").tobytes())
            else:
                f.write(struct.pack("<IIII", L[0], L[1], L[1], F32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.25, random_state=42, stratify=d.target)
    sizes = [X.shape[1], *args.hidden, 10]
    Ws, bs = train(Xtr, ytr, sizes)
    logits = forward(Xte, Ws, bs).astype(np.float32)
    print(f"architecture {sizes}, numpy test accuracy {(logits.argmax(1) == yte).mean():.4f}")

    write_model(os.path.join(args.out, "digits_f32.tinf"), Ws, bs, F32)
    write_model(os.path.join(args.out, "digits_i8.tinf"), Ws, bs, I8)
    with open(os.path.join(args.out, "digits_test.bin"), "wb") as f:
        f.write(struct.pack("<II", *Xte.shape))
        f.write(Xte.astype("<f4").tobytes())
        f.write(yte.astype("<u4").tobytes())
        f.write(logits.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
