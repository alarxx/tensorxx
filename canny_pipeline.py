import tensorx
from tensorx.opencv_utils import imshow
import numpy as np
import cv2

if __name__ == "__main__":
    print("Hello, TensorX!")

    path = "./lenna.png"
    t = tensorx.imread(path)

    print(f"t: rank({t.rank}), size({t.length})")
    print(f"t: dims({t.dims})")

    blurred = tensorx.gaussian_blur(t, 1)
    sobel = tensorx.sobel_operator(blurred)
    nms = tensorx.non_max_suppression(sobel)
    strongweak = tensorx.double_threshold(nms, 20.0, 80.0)
    chained = tensorx.hysterisis(nms, 20.0, 80.0)

    imshow("Original", t)
    imshow("Gaussian Blur", blurred)
    imshow("Sobel (norm)", sobel)
    imshow("NMS (norm)", nms)
    imshow("Double Threshold", strongweak)
    imshow("Hysteresis", chained)
