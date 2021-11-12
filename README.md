# SCAPT-ABSA

Code for EMNLP2021 paper: ["Learning Implicit Sentiment in Aspect-based Sentiment Analysis with Supervised Contrastive Pre-Training"](https://aclanthology.org/2021.emnlp-main.22.pdf)

## Overview

In this repository, we provide code for **S**uperived **C**ontr**A**stive **P**re-**T**raining (SCAPT) and aspect-aware fine-tuning, retrieved sentiment corpora from [YELP](https://www.yelp.com/dataset)/[Amazon](https://nijianmo.github.io/amazon/index.html) reviews, and [SemEval2014 Restaurant/Laptop](https://aclanthology.org/S14-2004.pdf) with addtional `implicit_sentiment` labeling.

SCAPT aims to tackle implicit sentiments expression in aspect-based sentiment analysis(ABSA).
In our work, we define implicit sentiment as sentiment expressions that contain no polarity markers but still convey clear human-aware sentiment polarity.

Here are examples for explicit and implicit sentiment in ABSA:

<div align="center"> <img src="https://user-images.githubusercontent.com/18156002/141302386-f01278fd-7a3d-4aa5-a6fc-fff99b649530.png" alt="examples" width="60%" /></div>

### SCAPT

SCAPT gives an aligned representation of sentiment expressions with the same sentiment label, which consists of three objectives:

- Supervised Contrastive Learning (SCL)
- Review Reconstruction (RR)
- Masked Aspect Prediction (MAP)

<div align="center"> <img src="https://user-images.githubusercontent.com/18156002/141302933-eb644d3d-cc5a-4ce9-8059-81a01395c366.png" alt="SCAPT" width="95%" /></div>

### Aspect-aware Fine-tuning

Sentiment representation and aspect-based representation are taken into account for sentiment prediction in aspect-aware fine-tuning.

<div align="center"> <img src="https://user-images.githubusercontent.com/18156002/141303033-c5b32e3c-5bf0-469b-ad72-0a0212a992ba.png" alt="Aspect_fine-tuning" width="50%" /></div>

## Requirement

- cuda 11.0
- python 3.7.9
  - lxml 4.6.2
  - numpy 1.19.2
  - pytorch 1.8.0
  - pyyaml 5.3.1
  - tqdm 4.55.0
  - transformers 4.2.2

## Data Preparation & Preprocessing

### For Pre-training

Retrieved sentiment corpora contain millions-level reviews, we provide download links for original corpora and preprocessed data.
Download if you want to do pre-training and further use them:

| File | Google Drive Link | Baidu Wangpan Link | Baidu Wangpan Code |
| :--: | :--: | :--: | :--: |
| scapt_yelp_json.zip | [link](https://drive.google.com/file/d/1C8BKQoYy24UNt46jGFuZxpPRH09teidv/view?usp=sharing) | [link](https://pan.baidu.com/s/17aSn3l4KqkWxHnb1IwqOTw) | q7fs |
| scapt_amazon_json.zip | [link](https://drive.google.com/file/d/1Z0OtxDa3pZ9oYigFjHaA35dugbRJyOJi/view?usp=sharing) | [link](https://pan.baidu.com/s/1BqbhPP3EpM2ddX59ZJ4q5Q) | i1da |
| scapt_yelp_pkl.zip | [link](https://drive.google.com/file/d/14omB11atNl1k6G66Du74kH9Qy2_z5yZO/view?usp=sharing) | [link](https://pan.baidu.com/s/1IIQHTIsXjhNYDTXa3vHs0Q) | j9ce |
| scapt_amazon_pkl.zip | [link](https://drive.google.com/file/d/1Vwr3SN4nl0uC3rpt2O4hrd7DJ5jDG0U0/view?usp=sharing) | [link](https://pan.baidu.com/s/1ezhzs1Mmr0clL_7bJBgNyw) | 3b8t |

These pickle files can also be generated from json files by the preprocessing method:

```bash
bash preprocess.py --pretrain
```

### For Fine-tuning

We have already combined the opinion term labeling to the original SemEval2014 datasets. For example:

```xml
    <sentence id="1634">
        <text>The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.</text>
        <aspectTerms>
            <aspectTerm term="food" polarity="positive" from="4" to="8" implicit_sentiment="False" opinion_words="exceptional"/>
            <aspectTerm term="kitchen" polarity="positive" from="55" to="62" implicit_sentiment="False" opinion_words="capable"/>
            <aspectTerm term="menu" polarity="neutral" from="141" to="145" implicit_sentiment="True"/>
        </aspectTerms>
        <aspectCategories>
            <aspectCategory category="food" polarity="positive"/>
        </aspectCategories>
    </sentence>
```

`implicit_sentiment` indicates whether it is an implicit sentiment expression and yield `opinion_words` if not implicit.
The `opinion_words` lebaling is credited to [TOWE](https://github.com/NJUNLP/TOWE).

Both original and extended fine-tuning data and preprocessed dumps are uploaded to this repository.

Consequently, the structure of your `data` directory should be:

```plain
├── Amazon
│   ├── amazon_laptops.json
│   └── amazon_laptops_preprocess_pretrain.pkl
├── laptops
│   ├── Laptops_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl
│   ├── Laptops_Test_Gold_Implicit_Labeled.xml
│   ├── Laptops_Test_Gold.xml
│   ├── Laptops_Train_v2_Implicit_Labeled_preprocess_finetune.pkl
│   ├── Laptops_Train_v2_Implicit_Labeled.xml
│   └── Laptops_Train_v2.xml
├── MAMS
│   ├── test_preprocess_finetune.pkl
│   ├── test.xml
│   ├── train_preprocess_finetune.pkl
│   ├── train.xml
│   ├── val_preprocess_finetune.pkl
│   └── val.xml
├── restaurants
│   ├── Restaurants_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl
│   ├── Restaurants_Test_Gold_Implicit_Labeled.xml
│   ├── Restaurants_Test_Gold.xml
│   ├── Restaurants_Train_v2_Implicit_Labeled_preprocess_finetune.pkl
│   ├── Restaurants_Train_v2_Implicit_Labeled.xml
│   └── Restaurants_Train_v2.xml
└── YELP
    ├── yelp_restaurants.json
    └── yelp_restaurants_preprocess_pretrain.pkl
```

## Pre-training

The pre-training is conducted on multiple GPUs.

- Pre-training [TransEnc|BERT] on [YELP|Amazon]:
  
  ```bash
  python -m torch.distributed.launch --nproc_per_node=${THE_CARD_NUM_YOU_HAVE} multi_card_train.py --config config/[yelp|amazon]_[TransEnc|BERT]_pretrain.yml
  ```

Model checkpoints are saved in `results`.

## Fine-tuning

- Directly train [TransEnc|BERT] on [Restaurants|Laptops|MAMS] As [TransEncAsp|BERTAsp]:

  ```bash
  python train.py --config config/[restaurants|laptops|mams]_[TransEnc|BERT]_finetune.yml
  ```

- Fine-tune the pre-trained [TransEnc|BERT] on [Restaurants|Laptops|MAMS] As [TransEncAsp+SCAPT|BERTAsp+SCAPT]:

  ```bash
  python train.py --config config/[restaurants|laptops|mams]_[TransEnc|BERT]_finetune.yml --checkpoint PATH/TO/MODEL_CHECKPOINT
  ```

Model checkpoints are saved in `results`.

## Evaluation

- Evaluate [TransEnc|BERT]-based model on [Restaurants|Laptops|MAMS] dataset:

  ```bash
  python evaluate.py --config config/[restaurants|laptops|mams]_[TransEnc|BERT]_finetune.yml --checkpoint PATH/TO/MODEL_CHECKPOINT
  ```

Our model parameters:

| Model | Dataset | File | Google Drive Link | Baidu Wangpan Link | Baidu Wangpan Code |
| :--: | :--: | :--: | :--: | :--: | :--: |
| TransEncAsp+SCAPT | SemEval2014 Restaurant | TransEnc_restaurants.zip | [link](https://drive.google.com/file/d/1ytYNtTKDcLK5bl1iBh2DnlhRba7UA6mi/view?usp=sharing) | [link](https://pan.baidu.com/s/1QLd4NStgk6OOk9MPrcM9xA) | 5e5c |
| TransEncAsp+SCAPT | SemEval2014 Laptop | TransEnc_laptops.zip | [link](https://drive.google.com/file/d/1VnVvHcFdTx5605VudYOYXzvidTnr8qwH/view?usp=sharing) | [link](https://pan.baidu.com/s/1DXKuCRjIViPwvNHUDpPbcA) | 8amq |
| TransEncAsp+SCAPT | MAMS | TransEnc_MAMS.zip | [link](https://drive.google.com/file/d/1_tWiQMjtn1mmXf3cOveqHVG9XhajM0Fb/view?usp=sharing) | [link](https://pan.baidu.com/s/1XUAnQS0iuhCWHFiN_Atw8g) | bf2x |
| BERTAsp+SCAPT | SemEval2014 Restaurant | BERT_restaurants.zip | [link](https://drive.google.com/file/d/1UGaTSdmHeIoYmFeVTE8ROvR-OFAY5Uo9/view?usp=sharing) | [link](https://pan.baidu.com/s/142hzJAYwvwISJ3ky_LmTUg) | 1w2e |
| BERTAsp+SCAPT | SemEval2014 Laptop | BERT_laptops.zip | [link](https://drive.google.com/file/d/1ouePhdk41Fsp3Ht9MmOLlv0k3dJbPqUg/view?usp=sharing) | [link](https://pan.baidu.com/s/1dtOOI0d7E48GnjcbZ9ZCVA) | zhte |
| BERTAsp+SCAPT | MAMS | BERT_MAMS.zip | [link](https://drive.google.com/file/d/1fje7c_MTcpauKiXfvPG9XrYYN_Xz4FIR/view?usp=sharing) | [link](https://pan.baidu.com/s/1tR7jsllJ_pOddFLdItNEeg) | 1iva |

## Citation

If you found this repository useful, please [cite](https://aclanthology.org/2021.emnlp-main.22.bib) our paper:

```plain
@inproceedings{li-etal-2021-learning-implicit,
    title = "Learning Implicit Sentiment in Aspect-based Sentiment Analysis with Supervised Contrastive Pre-Training",
    author = "Li, Zhengyan  and
      Zou, Yicheng  and
      Zhang, Chong  and
      Zhang, Qi  and
      Wei, Zhongyu",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.22",
    pages = "246--256",
    abstract = "Aspect-based sentiment analysis aims to identify the sentiment polarity of a specific aspect in product reviews. We notice that about 30{\%} of reviews do not contain obvious opinion words, but still convey clear human-aware sentiment orientation, which is known as implicit sentiment. However, recent neural network-based approaches paid little attention to implicit sentiment entailed in the reviews. To overcome this issue, we adopt Supervised Contrastive Pre-training on large-scale sentiment-annotated corpora retrieved from in-domain language resources. By aligning the representation of implicit sentiment expressions to those with the same sentiment label, the pre-training process leads to better capture of both implicit and explicit sentiment orientation towards aspects in reviews. Experimental results show that our method achieves state-of-the-art performance on SemEval2014 benchmarks, and comprehensive analysis validates its effectiveness on learning implicit sentiment.",
}
```
