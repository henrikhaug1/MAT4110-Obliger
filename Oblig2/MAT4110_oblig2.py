import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def convert_grayscale(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


def svd_compress(im: np.ndarray, r: int) -> np.ndarray:
    U, S, VT = np.linalg.svd(im, full_matrices=False)
    r = int(min(r, S.shape[0]))
    im_r = (U[:, :r] * S[:r]) @ VT[:r, :]
    return np.clip(im_r, 0.0, 1.0)


def compression_ratio(m: int, n: int, r: int) -> float:
    return (r * (m + n + 1)) / (m * n)


paths = ["board.png", "jellyfish.jpg", "outdoors.jpg"]
images = [convert_grayscale(p) for p in paths]
r_list = [5, 20, 50, 100]

for idx, im in enumerate(images):
    # ---------- Compute SVD ----------
    U, S, VT = np.linalg.svd(im, full_matrices=False)

    # ---------- Plot log of singular values ----------
    plt.figure(figsize=(6, 4))
    plt.plot(np.log10(S + 1e-12), marker="o", markersize=3)
    plt.xlabel("Index k")
    plt.ylabel(r"$log10(\sigma_k)$")
    plt.title(f"Log of singular values (Image {idx + 1})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Compress and show results ----------
    m, n = im.shape
    plt.figure(figsize=(12, 2.5))
    plt.subplot(1, len(r_list) + 1, 1)
    plt.imshow(im, cmap="gray")
    plt.axis("off")
    plt.title("original")

    for j, r in enumerate(r_list, start=2):
        im_r = (U[:, :r] * S[:r]) @ VT[:r, :]
        im_r = np.clip(im_r, 0.0, 1.0)
        cr = compression_ratio(m, n, r)
        plt.subplot(1, len(r_list) + 1, j)
        plt.imshow(im_r, cmap="gray")
        plt.axis("off")
        plt.title(f"r={r}\nCR={cr:.2f}")

    plt.suptitle(f"Image {idx + 1} (m={m}, n={n})")
    plt.tight_layout()
    plt.show()
