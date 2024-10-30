[GhostNetV2: Enhance Cheap Operation with Long-Range Attention](https://arxiv.org/abs/2211.12905)

### Installation

```
conda create -n PyTorch python=3.10.10
conda activate PyTorch
conda install python=3.10.10 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
pip install pyyaml
pip install timm
pip install tqdm
```

### Note

* The test results including accuracy, params and FLOP are obtained by using fused model

### Train

* Configure your `IMAGENET` dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your `IMAGENET` path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

|     Version     | Epochs | Top-1 Acc | Top-5 Acc | Params (M) | FLOP (M) |                    Download |
|:---------------:|:------:|----------:|----------:|-----------:|---------:|----------------------------:|
| GhostNetV2-1.0  |  450   |         - |         - |      6.126 |  167.689 |                           - |
| GhostNetV2-1.0* |  450   |     75.15 |     92.25 |      6.126 |  167.689 | [model](./weights/v2_10.pt) |
| GhostNetV2-1.3* |  450   |     76.67 |     93.32 |      8.920 |  270.156 | [model](./weights/v2_13.pt) |
| GhostNetV2-1.6* |  450   |     77.76 |     93.97 |     12.343 |  399.636 | [model](./weights/v2_16.pt) |

* `*` means that weights are ported from original repo, see reference

#### Reference

* https://github.com/huawei-noah/Efficient-AI-Backbones
