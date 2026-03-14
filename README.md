# Tensor-library (Python)

Python bindings for the [Tensor-library (C++)](https://github.com/alarxx/Tensor-library).

- C++20
- CMake
- Python3
- pybind11
- scikit-build-core
- cibuildwheel
- twine

---

## Example

Create virtual environment:
```sh
python3 -m venv .venv
. .venv/bin/activate
```
Install with pip:
```sh
pip install tensorxx
```

---

Tensor creation and access example:
```python
import tensorxx
print(tensorxx)

import tensorxx._tensorxx
print(tensorxx._tensorxx)

if __name__ == "__main__":
    print("add_ints:", tensorxx.add_ints(2, 3))

    print("make_tensor:")
    tensorxx.make_tensor()

    t0 = tensorxx.Tensor()
    print("t0.rank:", t0.rank, "t0.dims:", t0.dims, "t0.length:", t0.length)
    print("t0:", t0)

    s = tensorxx.scalar(3.14)
    print("s:", s)
    print("s.rank:", s.rank, "s.dims:", s.dims, "s.length:", s.length)
    print("s.get:", s.get())     # 3.14
    s.set(2.71)
    print("s.set(2.71)->s.get:", s.get())     # 2.71

    t1 = tensorxx.Tensor(2, (2, 3))
    print("t1.rank:", t1.rank, "t1.dims:", t1.dims, "t1.length:", t1.length)
    print("t1:", t1)

    t2 = tensorxx.Tensor(2, 3, 4)
    print("t2.rank:", t2.rank, "t2.dims:", t2.dims, "t2.length:", t2.length)
    print("t2:", t2)

    v = tensorxx.from_list([1, 2, 3])
    print("v:", v)

    m = tensorxx.from_list([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ])
    print("m:", m)
    print("m.rank:", m.rank, "m.dims:", m.dims, "m.length:", m.length)
    print("m.get(1, 1):", m.get(1, 1))
```

---

Image processing example:
```sh
import tensorxx
from tensorxx.opencv_utils import imshow
import numpy as np
import cv2

if __name__ == "__main__":
    print("Hello, TensorX!")

    path = "./lenna.png"
    t = tensorxx.imread(path)

    print(f"t: rank({t.rank}), size({t.length})")
    print(f"t: dims({t.dims})")

    blurred = tensorxx.gaussian_blur(t, 1)
    sobel = tensorxx.sobel_operator(blurred)
    nms = tensorxx.non_max_suppression(sobel)
    strongweak = tensorxx.double_threshold(nms, 20.0, 80.0)
    chained = tensorxx.hysterisis(nms, 20.0, 80.0)

    imshow("Original", t)
    imshow("Gaussian Blur", blurred)
    imshow("Sobel (norm)", sobel)
    imshow("NMS (norm)", nms)
    imshow("Double Threshold", strongweak)
    imshow("Hysteresis", chained)
```

---

## Build

Add Tensor-library:
```sh
git submodule add -b cxx-mappings git@github.com:alarxx/Tensor-library.git
```

---

Install Docker:
Root:
```sh
apt install docker.io
sudo systemctl enable docker # autostart
sudo systemctl start docker
sudo systemctl status docker
sudo usermod -aG docker $USER
```
User:
```sh
su -c "sudo usermod -aG docker $USER"
newgrp docker
docker version
docker ps
```

---


```sh
apt install python3 python3-full python3-pip python3-venv
#sudo apt install python3.8
python3 --version

python3 -m venv .venv
. .venv/bin/activate

python -m pip install -U pip build
python -m pip install -U pybind11 cmake ninja
python -m pip install -U twine # upload
python -m pip install -U scikit-build-core # hatchling
python -m pip install -U cibuildwheel # cross-platform build
```

---

Build with **hatchling** vs. **scikit_build_core** (CMake).

Generating distribution archives from `pyproject.toml` file with configuration metadata:
```sh
python -m build
```
- `.tar.gz` - source distribution
- `.whl` - built distribution

---

Try to install locally:
```sh
python -m pip install --force-reinstall dist/*.whl
```
- `--force-reinstall` - if version didn't change

Uninstall local tensorxx:
```sh
python -m pip uninstall tensorxx
```

Inside tensorxx/ should be shared libraries (.so)
```sh
unzip -l dist/*.whl
```

---

Cross-platform build:
```sh
cibuildwheel --output-dir dist
```

Upload to PyPI:
```sh
python -m twine upload dist/*
```


Разобраться с manylinux:
```sh
export CIBW_REPAIR_WHEEL_COMMAND_LINUX='LD_LIBRARY_PATH=/opt/opencv/lib64 auditwheel repair -w {dest_dir} {wheel}'


export CIBW_BEFORE_BUILD_LINUX='set -eux; find /opt/opencv -maxdepth 3 -type f -name "libopencv_core.so*" -o -name "libopencv_imgproc.so*" -o -name "libopencv_imgcodecs.so*" -o -name "libopencv_highgui.so*" || true; find /opt/opencv -maxdepth 3 -type d -print;'

export CIBW_SKIP="*-musllinux_*"
```
