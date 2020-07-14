## Progressive Unsupervised Person Re-identification by Tracklet Association with Spatio-Temporal Regularization [[link](https://ieeexplore.ieee.org/abstract/document/9057713)]

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

Download DukeMTMC-reID dataset then extract file to your data directory ( denoted as `${data_root}` ). The file structure is as follows

```bash
${data_root}
└── dukemtmc-reid/
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
modify the `${data_root}` in `./scripts/train_dukemtmcreid.sh`, then run

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