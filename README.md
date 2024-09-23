# [CoRL24] Contrastive &lambda;-Repformer: Task Success Prediction for Open-Vocabulary Manipulation Based on Multi-Level Aligned Representation

- Accepted at CoRL 2024
- [Project Page](https://5ei74r0.github.io/contrastive-lambda-repformer.page/)
- arXiv
- [Dataset](https://contrastive-lambda-repformer.s3.amazonaws.com/dataset/dataset.tar.gz)
- Project Video

## Abstract

In this study, we consider the problem of predicting task success for open-vocabulary manipulation by a manipulator, based on instruction sentences and egocentric images before and after manipulation. Conventional approaches, including multimodal large language models (MLLMs), often fail to appropriately understand detailed characteristics of objects and/or subtle changes in the position of objects. We propose Contrastive &lambda;-Repformer, which predicts task success for table-top manipulation tasks by aligning images with instruction sentences. Our method integrates the following three key types of features into a multi-level aligned representation: features that preserve local image information; features aligned with natural language; and features structured through natural language. This allows the model to focus on important changes by looking at the differences in the representation between two images. We evaluate Contrastive &lambda;-Repformer on a dataset based on a large-scale standard dataset, the RT-1 dataset, and on a physical robot platform. The results show that our approach outperformed existing approaches including MLLMs. Our best model achieved an improvement of 8.66 points in accuracy compared to the representative MLLM-based model.

## Instructions

We assume the following environment for our experiments:

- Python 3.8.10
- PyTorch version 2.1.0 with CUDA 11.7 support

### Clone & Install

```bash
cd Contrastive_Lambda-Repformer
```

```bash
pyenv virtualenv 3.8.10 contrastive_lambda-repformer
pyenv local contrastive_lambda-repformer
pip install -r requirements.txt
```

### Datasets

- Our dataset can be downloaded at [this link](https://contrastive-lambda-repformer.s3.amazonaws.com/dataset/dataset.tar.gz).
  - Unzip and extract the `data` using the following commands.

```bash
mkdir data
cd data
wget https://contrastive-lambda-repformer.s3.amazonaws.com/dataset/dataset.tar.gz
tar -xvzf dataset.tar.gz
cd ..
```

### Extract Features

```bash
python src/utils/extract_ViT_features.py
python src/utils/extract_DINOv2_features.py
python src/utils/retrieve_InstructBLIP_narratives.py
```

### Train & Evaluation on SP-RT-1 Dataset

- Please note that the first run of the program may take some time to complete, as the remaining features are extracted.
- Previously saved features are used in subsequent runs for faster execution.

```bash
export PYTHONPATH=`pwd`
export OPENAI_API_KEY="Your OpenAI API Key"
python src/main.py
```

### Evaluation on SP-RT-1 Dataset

If you already have a valid checkpoint, you can evaluate the model on the dataset.

1. Set checkpoint_path in configs/config.json to the relative path of your checkpoint file.
2. Execute the following commands.
```bash
export PYTHONPATH=`pwd`
export OPENAI_API_KEY="Your OpenAI API Key"
python src/test_model.py
```

### Evaluation on Other Datasets (SP-HSR dataset)

THe model can also be evaluated on other datasets, such as the SP-RT-1 dataset.
Using checkpoints at around 50 epochs is recommended.

1. Set dataset_name in configs/config.json to "SP-HSR".
2. Run the commands in "Extract Features".
3. Run the commands in "Evaluation on SP-RT-1 Dataset".

## Bibtex

```
@inproceedings{
    goko2024task,
    title     = {{Task Success Prediction for Open-Vocabulary Manipulation Based on Multi-Level Aligned Representations}},
    author    = {Goko, Miyu and Kambara, Motonari and Saito, Daichi and Otsuki, Seitaro and Sugiura, Komei},
    booktitle = {8th Annual Conference on Robot Learning},
    year      = {2024}
}
```

## License
This work in licensed under the MIT license. To view a copy of this license, see [LICENSE](LICENSE).
