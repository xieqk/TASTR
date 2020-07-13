## Progressive Unsupervised Person Re-identification by Tracklet Association with Spatio-Temporal Regularization

The implementation code and models of TASTR.

### Environment

- Ubuntu 16.04
- Python 3.7
- Pytorch 1.0.1
- CUDA 9.0
- NVIDIA GTX 1080ti x 2

### Installation

```bash
git clone https://github.com/xieqk/TASTR.git
```

### Prepare dataset

Download DukeMTMC-reID dataset then extract file. The file structure is as follows

```bash
dukemtmc-reid/
└── DukeMTMC-reID
    ├── bounding_box_test
    ├── bounding_box_train
    ├── CITATION.txt
    ├── LICENSE_DukeMTMC-reID.txt
    ├── LICENSE_DukeMTMC.txt
    ├── query
    └── README.md
```

### Training

```bash
./scripts/train_dukemtmcreid.sh
```

### Citation
```latex
@article{xie2020progressive,
  title={Progressive Unsupervised Person Re-identification by Tracklet Association with Spatio-Temporal Regularization},
  author={Xie, Qiaokang and Zhou, Wengang and Qi, Guo-Jun and Tian, Qi and Li, Houqiang},
  journal={IEEE Transactions on Multimedia (TMM)},
  year={2020},
  publisher={IEEE}
}
```



Request the Campus4K dataset from xieqiaok [at] mail.ustc.edu.cn. 