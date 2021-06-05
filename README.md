# Model Extraction and Adversarial Transferability, Your BERT is Vulnerable!

## Descriptions
This repo contains source code and pre-processed corpora for "**Model Extraction and Adversarial Transferability, Your BERT is Vulnerable!**" (accepted to NAACL-HLT 2021)


## Dependencies
* python3
* pytorch>=1.4
* transformers==3.0.2
* cuda 10.0

## Data
Please download data from [here](https://drive.google.com/file/d/1WPg7ufEmZ-1zASsSa-2ctC2PiqIfXN0F/view?usp=sharing)

## Usage
```shell
git clone https://github.com/xlhex/extract_and_transfer.git
```

## Train a victim model
```shell
TASK=blog
SEED=1234
sh train_vic.sh $TASK $SEED
```

## Query the victim model
```shell
TASK=blog
SEED=1234
QUERY_FILE=review_sample.tsv
PRED_FILE=review_pred.tsv
DEFENSE=temp # temp or perturb
sh pred.sh $TASK $SEED $QUERY_FILE $PRED_FILE $DEFENSE

python construct_distilled_data.py data/$TASK/review_sample.tsv data/$TASK/review_pred.tsv data/$TASK/review_train.tsv
```

## Train an extracted model
```shell
TASK=blog
SEED=1234
sh train_extract.sh $TASK $SEED
```

## Citation
Please cite as:

```bibtex
@inproceedings{he2021model,
  title={Model Extraction and Adversarial Transferability, Your BERT is Vulnerable!},
  author={He, Xuanli and Lyu, Lingjuan and Sun, Lichao and Xu, Qiongkai},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={2006--2012},
  year={2021}
}
```
