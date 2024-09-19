# [CoRL24] Contrastive λ-Repformer: Task Success Prediction for Open-Vocabulary Manipulation Based on Multi-Level Aligned Representation

- Accepted at CoRL 2024
- [Project Page](https://lambda-repformer-project-pa-eziy1.kinsta.page/)
- arXiv
- [Dataset](https://contrastive-lambda-rcepformer.s3.amazonaws.com/dataset/dataset.tar.gz)

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
  - Unzip and extract the `data`.


### Train & Evaluation on SP-RT-1 dataset

```bash
export PYTHONPATH=`pwd`
export OPENAI_API_KEY="Your OpenAI API Key"
python src/main.py
```
