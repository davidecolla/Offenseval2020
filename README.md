## Overview
This repository contains the source code for the models used for _OffensEval 2_ team's submission
for [SemEval-2020 Task 12 “OffensEval 2: Multilingual Offensive Language Identification in Social Media”](https://sites.google.com/site/offensevalsharedtask/).
The model is described in the paper 
["GruPaTo at SemEval-2020 Task 12: Retraining mBERT on Social Media and Fine-tune Offensive Language Models"](offenseval_2020.pdf).


## Prerequisites
#### 1 - Install Requirements
```
conda env create -f environment.yml
```
NB. It requires [Anaconda](https://www.anaconda.com/distribution/)

#### 2 - Download pre-trained models and fine-tuned models
Download data from the following links and unpack the archive in the data folder.
- [EN - submitted run (mBERT-E3)](https://drive.google.com/drive/folders/1gpZgekt4L1p0yR-e-wIzflb779aBl-S9?usp=sharing)
- [DA - submitted run (mBERT-D3)](https://drive.google.com/drive/folders/1gfzaO104cQh-AlrKTVs-OvTXdinhMVcU?usp=sharing) 
- [TR - submitted run (mBERT-T1)](https://drive.google.com/drive/folders/1gfzaO104cQh-AlrKTVs-OvTXdinhMVcU?usp=sharing)

## Execution
In order to produce the submission files run the script 'code/text_classification_with_pretrained_model.py'. The script takes three arguments:
1. The directory where the fine-tuned model has been saved
2. The directory where the output file will be saved
3. The input test file, the one provided by the Offenseval 2 organizers

#### Run on English
```
python text_classification_with_pretrained_model.py ../data/en/c/finetuned_text_classification ../data/en/c/finetuned_text_classification/eval ../data/en/offenseval-en-testset/test_a_tweets.tsv
```

#### Run on Turkish
```
python text_classification_with_pretrained_model.py ../data/tr/e/finetuned_text_classification ../data/tr/e/finetuned_text_classification/eval ../data/tr/offenseval-tr-testset/offenseval-tr-testset-v1.tsv
```

#### Run on Danish
```
python text_classification_with_pretrained_model.py ../data/da/transfer/finetuned_text_classification ../data/da/transfer/finetuned_text_classification/eval ../data/da/offenseval-da-testset/offenseval-da-test-v1-nolabels.tsv
```


## Starting from scratch
#### 1 - Pre-train language model
In order to pre-train the language model we exploited the [HuggingFace example](https://github.com/huggingface/transformers/tree/master/examples#language-model-training) for language model training.
The pre-training has to be done by direcly executing the Language Model Pre-training example from the repository:
```
git clone https://github.com/huggingface/transformers
```

Once the repository has been cloned, run the 'run_language_modeling.py' script, eg. for the English language:
```
python run_language_modeling.py --output_dir=../data/en/c/pretrained_lm --model_type=bert --model_name_or_path=bert-base-multilingual-cased --do_train --train_data_file=../data/en/c/twitter_data_en.txt --mlm
```

#### 2 - Fine-tune language model for classification
Once the pre-training of the model has been completed the fine-tuning can be performed by running the 'code/fine_tune_language_model_no_validation.py' script.
The script takes three arguments:
1. The directory where the pre-trained model has been saved
2. The directory where the fine-tuned model will be saved
3. The input fine-tuning file

Eg. for the english language:
```
python fine_tune_language_model_no_validation.py ../data/en/c/pretrained_lm ../data/en/c/finetuned_text_classification ../data/en/fine_tuning-data/training.tsv
```

In order to produce the Offenseval 2 submission files run the **Execution** step. 
