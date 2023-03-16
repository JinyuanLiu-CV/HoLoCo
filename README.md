# HoLoCo
Offical Code for: Jinyuan Liu, Guanyao Wu, Junsheng Luan, Zhiying Jiang, Risheng Liu, Xin Fan*,**“HoLoCo: Holistic and Local Contrastive Learning Network for Multi-exposure Image Fusion”**, Information Fusion[J], 2023.


### Set Up on Your Own Machine

#### Virtual Environment

We strongly recommend that you use Conda as a package manager.

```shell
# create virtual environment
conda create -n holoco python=3.10
conda activate holoco
# select and install pytorch version yourself
# install requirements package
pip install -r requirements.txt
```

#### Download Checkpoints

Before testing or training HoLoCo, we strongly recommend downloading the following pre-trained model and placing them in the **./checkpoints** folder

- [Google Drive](https://drive.google.com/drive/folders/1sOp9Fmtfm_U4w3_-pGWyuYOeHC__buoT?usp=sharing)
- [Baidu Yun](https://pan.baidu.com/s/14Uo_0RfiPBc2NPEaggwgLw?pwd=HLCo)

## Test / Train
```shell
# Test: use given example and save fused color images to result/SICE
# If you want to test the custom data, please modify the file path in **test.py**
python start_test.py

# Train: 
# Please prepare the custom data and change the modifiable options in **start_train.py** (optional)
python start_train.py
```

## Citation

If this work has been helpful to you, we would be appreciate if you could cite our paper!

```
@article{liu2023holoco,
  title={HoLoCo: Holistic and local contrastive learning network for multi-exposure image fusion},
  author={Liu, Jinyuan and Wu, Guanyao and Luan, Junsheng and Jiang, Zhiying and Liu, Risheng and Fan, Xin},
  journal={Information Fusion},
  year={2023},
  publisher={Elsevier}
}
```
