# SAM-for-Medical-Image-Segmentation
Course project of Machine Learning (PKU 2023 Autumn)

# Quick Start
1. Install all the requirements with `pip install -r requirements.txt`
2. Install [SAM](https://github.com/facebookresearch/segment-anything/tree/main)
3. Download [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752) dataset.
4. Each task has a separate folder named `task*` (* is 1/2/3), modify path to BTCV dataset in config file `config.yaml`.  
5. Run `main.py` in folder `code/task*`
6. We provide visulization tool as `code/task3/visulize3D.py`

# Model Zoo
Here we provide pretrained classifier and fintuned SAM model with best mDice score(prompt type is 'bbox')
1. [Finetuned SAM]()
2. [Classifier]()

