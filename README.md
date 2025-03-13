# face_detecteion_dev

This repo is for face detection development. It supports traingin, evaluation, and conversion to onnx.
SCRFD is the first implementation of this repo.

## Environment setup

### MacOS

- [Pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#macos)

- Set up `xz` for `pyenv` to install Python 3.10.16

  ```bash
  brew install xz
  echo 'export PYTHON_CONFIGURE_OPTS="--with-lzma=$(brew --prefix xz)' > ~/.zshrc
  exec $SHELL
  pyenv install 3.10.16
  pyenv virtualenv 3.10.16 face_detection_dev
  pyenv activate face_detection_dev
  ```

- Python packages

  ```bash
  pip install wheel
  pip install torch==2.4.1 torchvision # must fix for mps bug, see https://github.com/pytorch/pytorch/issues/142344
  MAX_JOBS=4 MMCV_WITH_OPS=1 pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0
  pip install -r docker/requirements_macos.txt
  ```

### Linux

- [Pyenv](https://github.com/pyenv/pyenv-installer)

  ```bash
  pyenv install 3.10.16
  pyenv virtualenv 3.10.16 face_detection_dev
  pyenv activate face_detection_dev
  ```

- Python packages

  ```bash
  pip install wheel
  pip install torch==2.5.1 torchvision
  MAX_JOBS=4 MMCV_WITH_OPS=1 pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0
  pip install -r docker/requirements_linux.txt
  ```

### Docker

- Build Docker image

  ```bash
  cd docker
  bash build.sh
  cd -
  ```

- Run Docker container

  ```bash
  bash docker/run_docker.sh $CMD
  ```

## Train

### without docker

```bash
python -u train.py ${config}
```

### with docker

```bash
bash docker/run_docker.sh python train.py ${config}
```

## To onnx

```bash
ipython -- torch2onnx.py --config ${config} --ckpt_fpath ${ckpt_fpath} --onnx_fpath ${onnx_fpath}
```

## Benchmarks

### Widerface Evaluation

```bash
ipython -- benchmark/widerface.py --onnx_fpath ${onnx_fpath} --data_folder ${data_folder}
```

### NIST Evaluation

```bash
ipython -- benchmark/nist.py --onnx_fpath ${onnx_fpath} --data_folder ${data_folder}
```
