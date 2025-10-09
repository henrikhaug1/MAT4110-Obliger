import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------- Load & Convert Images ----------
im1 = np.array(Image.open("board.png").convert("L"))
im2 = np.array(Image.open("jellyfish.jpg").convert("L"))
im3 = np.array(Image.open("outdoors.jpg").convert("L"))

images = [im1 / 255.0, im2 / 255.0, im3 / 255.0]

# ---------- Plot regular B&W images ----------
# for im in images:
#     print(im.shape)
#     plt.figure()
#     plt.imshow(im, cmap="gray")

# plt.show()

# ---------- Calculate SVDs ----------


def svd_compress(im: np.ndarray, r: int):
    U, S, VT = np.linalg.svd(im, full_matrices=False)
    r = int(min(r, S.shape[0]))
    im_r = (U[:, :r] * S[:r]) @ VT[:r, :]
    im_r = np.clip(im_r, 0.0, 1.0)
    return im_r


for im in images:
    im = svd_compress(im, 1000)
    plt.imshow(im)
    plt.show()
