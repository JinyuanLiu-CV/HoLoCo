# HoLoCo
Official Code for: Jinyuan Liu, Guanyao Wu, Junsheng Luan, Zhiying Jiang, Risheng Liu, Xin Fan*,**“HoLoCo: Holistic and Local Contrastive Learning Network for Multi-exposure Image Fusion”**, Information Fusion[J], 2023.

- [*[Information Fusion]*](https://www.sciencedirect.com/science/article/pii/S1566253523000672)
- [*[Google Scholar]*](https://scholar.google.com.hk/scholar?as_sdt=0%2C5&q=HoLoCo%3A+Holistic+and+local+contrastive+learning+network+for+multi-exposure+image+fusion&btnG=)

## Preview of HoLoCo
---

![preview](assets/workflow.png)
<p align="center">
  <img src="assets/preview1.gif" width="100%">
</p>

 
---


## Set Up on Your Own Machine

### Virtual Environment

We strongly recommend that you use Conda as a package manager.

```shell
# create virtual environment
conda create -n holoco python=3.10
conda activate holoco
# select and install pytorch version yourself (Necessary & Important)
# install requirements package
pip install -r requirements.txt
```

### Download Checkpoints

Before testing or training HoLoCo, we strongly recommend downloading the following pre-trained model and placing them in **./checkpoints** folder.

- [Google Drive](https://drive.google.com/drive/folders/1sOp9Fmtfm_U4w3_-pGWyuYOeHC__buoT?usp=sharing)
- [Baidu Yun](https://pan.baidu.com/s/14Uo_0RfiPBc2NPEaggwgLw?pwd=HLCo)

### Test / Train
This code natively supports the same naming for over-/under-exposed image pairs. An naming example can be found in **./datasets/SICE** folder.
```shell
# Test: use given example and save fused color images to result/SICE
# If you want to test the custom data, please modify the file path in 'test.py'
python start_test.py

# Train: 
# Please prepare the custom data and change the modifiable options in 'start_train.py' (optional)
python start_train.py
```

## Citation

If this work has been helpful to you, we would appreciate it if you could cite our paper! 

```
@article{liu2023holoco,
  title={HoLoCo: Holistic and local contrastive learning network for multi-exposure image fusion},
  author={Liu, Jinyuan and Wu, Guanyao and Luan, Junsheng and Jiang, Zhiying and Liu, Risheng and Fan, Xin},
  journal={Information Fusion},
  year={2023},
  publisher={Elsevier}
}
```
