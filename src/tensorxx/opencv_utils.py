import numpy as np
import cv2

def tensor2numpy(t):
    h, w = t.dims
    out = np.empty((h, w), dtype=np.float32)

    getter = lambda r, c: t.get(r, c)

    for r in range(h):
        for c in range(w):
            out[r, c] = float(getter(r, c))

    return out.astype(np.uint8)

def imshow(title, tensor):
    img = tensor2numpy(tensor)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
