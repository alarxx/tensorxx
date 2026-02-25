# tensorx-python

Python bindings for the Tensor-library

- C++20
- CMake
- Python3
- pybind11
- scikit-build-core

---

Add Tensor-library:
```sh
git submodule add -b c++mappings git@github.com:alarxx/Tensor-library.git
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
python -m pip install --upgrade pip
python -m pip install --upgrade build
python -m pip install -U twine
python -m pip install -U cibuildwheel
```

---

Build with **hatchling** vs. **scikit_build_core** (CMake).

Generating distribution archives from `pyproject.toml` file with configuration metadata:
```sh
python -m build
```
- `.tar.gz` - source distribution
- `.whl` - built distribution

Try to install locally:
```sh
python -m pip install --force-reinstall dist/*.whl
```
- `--force-reinstall` - if version didn't change

Uninstall local tensorx-python:
```sh
python -m pip uninstall tensorx-python
```

Inside tensorx/ should be shared libraries (.so)
```sh
unzip -l dist/*.whl
```

---

Cross-platform build:
```sh
cibuildwheel --output-dir dis
```

Upload to PyPI:
```sh
python -m twine upload dist/*
```
