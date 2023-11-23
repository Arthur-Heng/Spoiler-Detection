## Detecting Spoilers in Movie Reviews with External Movie Knowledge and User Networks

paper link: [https://arxiv.org/abs/2304.11411](https://arxiv.org/abs/2304.11411)

### How to download the LCS dataset and UKM movie database
The LCS datset and UKM movie database is available at [Google Drive](https://drive.google.com/drive/folders/1By6_vmaOAaLnZGwrFCGb8UeP8eB2ae4L?usp=share_link).
Please apply for access by contacting wh2213210554@stu.xjtu.edu.cn with your institutional email address and clearly state your institution, your research advisor (if any), and your use case of the data.

We also provided preprocessed data for the Kaggle dataset, which can be directly download from [Google Drive](https://drive.google.com/file/d/1uN3utgi-puUzQKTCWj5UK4acalBzJH9M/view?usp=sharing).

### Environment
`conda env create -f environment.yml` would generate a conda environment called `spoiler` that should be able to run the code.

### How to run the code
First create the conda environment.
#### Step 0: download preprocessed data
Download preprocessed data for the Kaggle dataset from [Google Drive](https://drive.google.com/file/d/1uN3utgi-puUzQKTCWj5UK4acalBzJH9M/view?usp=sharing) to the repository directory and unzip it.

#### Step 1: cd to src and train the model
```
cd src
python train.py  --device <your_device_id>
```

### Citation 
If you find this dataset or codebase useful in your research, please cite the following paper.

```bibtex
@article{wang2023detecting,
  title={Detecting Spoilers in Movie Reviews with External Movie Knowledge and User Networks},
  author={Wang, Heng and Zhang, Wenqian and Bai, Yuyang and Tan, Zhaoxuan and Feng, Shangbin and Zheng, Qinghua and Luo, Minnan},
  journal={arXiv preprint arXiv:2304.11411},
  year={2023}
}
```
