# ELSA

ELSA is an emoji-powered representation learning framework for cross-lingual sentiment classification. 

The workflow of ELSA consists of the following phases:

1. It uses large-scale Tweets to learn word embeddings through Word2Vec of both the source and the target languages in an unsupervised way. 

2. In a distant-supervised way, it uses emojis as complementary sentiment labels to transform the word embeddings into a higher-level sentence representation that encodes rich sentiment information via an emoji-prediction task through an attention-based stacked bi-directional LSTM model. This step is also conducted separately for both the source and the target languages. 

3. It translates the labeled English data into the target language through an off-the-shelf machine translation system, represent the pseudo parallel texts with the pre-trained language-specific models, and use an attention model to train the final sentiment classifier.

You can see the WWW 2019 (know as The Web Conference) paper “**Emoji-Powered Representation Learning for Cross-Lingual Sentiment Classification**” for more details.

## Overview

- dataset/ contains the raw and processed data used for training and testing our approach. It contains two subfolders: 
  - Amazon review/ contains the processed amazon review dataset created by [Pretten-
    hofer and Stein et al.](http://www.aclweb.org/anthology/P10-1114). Aside from the given parallel reviews for Japanese, French and German to English, we translate English training and testing reviews to other languages through [Google Translate][https://translate.google.com]. Each line of the dataset file is composed of `sentiment label \t english version \t other language version`, please use json to load the according text of each review (they are already processed into list of words).
  - Tweets sentiment/ contains the collected tweets sentimental dataset proposed in [Deriu, Jan, et al.][https://github.com/spinningbytes/deep-mlsa]. Each file use the same format of the review file except for they normally contain only one sentence in tweets. For english sentimenal tweets, we collected from semeval compition in 2015 and 2016. You can refer to our paper for the detailed split info. 
- scripts/ contains the script for processing tweets, training word embedding and tidying emoji tweets into the numpy version for final ELSA model training.
  - process_raw_tweet/ simply modify `tweet_token.py` with `input_file, output_file, emoji_file` field to complete the tokenization, extraction emoji from tweets process. Set `JAPAN=True/False` in the `word_generator.py` file for preprocessing tweets in different languages. 
  - process_pre_tweets/  contains the vocabulary file that can be generated through scripts in `process_raw_tweet/ `. Please change the filename and path name in each scripts and run `tidy_wordvec.py, tidy_vocab.py, tidy_tweet_elsa.py` in sequence. 
- elsa/ contains the sentence level emoji prediction ELSA model and document level ELSA model. The configuration files are all in yaml.
- pretrained/ contains the sentence representation, elsa sentence models and elsa document models for English, Japanese, French, and German. 

## Setup

1. We assume that you're using Python 3.6 with pip installed. As a backend you need to install either Theano (version 0.9+) or Tensorflow (version 1.3+). To run the code, you need the following dependencies:

- [Keras](https://github.com/keras-team/keras) (above 2.0.0)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [text-unidecode](https://github.com/kmike/text-unidecode)
- [Mecab](http://taku910.github.io/mecab/) tokenize Japanese.
- [yaml](https://github.com/yaml)

You can use the python package manager of your choice (*pip/conda*) to install the dependencies.
The code is tested on the *Linux* operating system. 

2. To reproduce our main results, for `sentence level` ELSA model, you can change the configuration file `elsa_test.yaml` by setting the

   ``````
   cur_lan: "fr" or "jp" or "de"
   input_dir: /absolute path to processed Amazon review dataset.
   output_dir: /directory to store the sentence representations. 
   vocab_path: /directory that contains the vocabulary file of cur_lan.
   checkpoint_weight_path: /pretrained sentence embedding for chosen language.
   cur_en_or_ot: True and False, please set this field twice and run the script repectively.
   ``````

   Then run `python test_elsa_sentence.py`. 

   Or you can use the `pretrained sentence embedding ` in `pretrained/` and set `elsa_doc.yaml` as

   ```
   mode: 'test'
   cur_test: 'en_jp/' or 'en_fr/' or 'en_de/'
   pretrain_path: /absolute path to elsa_doc_model
   cur_cat: '_music' or '_dvd' or '_books'
   ```

   Then run `python elsa_doc.py`. You will have the results listing in the following format:

   `en_jp dvd Test Accuracy: 0.8045`

3. After detailed preprocessing of tweets and dataset decribed above, in order to train a new `sentence level` ELSA model, you can run the scripts in the elsa/ directory and change the `elsa_train.yaml` as you please. 

   Furthermore, to train a new `document level` ELSA model, after collecting the sentence representation for each sentence in the docuement, you can modify the `mode: 'train'`in `elsa_doc.yaml` file and finetune your own model accordingly.

## Dataset

We sadly cannot release our large-scale dataset of Tweets used to train representation learning models due to licensing restrictions.

We upload all the benchmark datasets to this repository for convenience. As they were not collected and released by us, we do not claim any rights on them. If you use any of these datasets, please make sure you fulfill the licenses that they were released with and consider citing the original authors.

### Citation

Please consider citing the following paper when using our code or pretrained models for your application.

```
@inproceedings{chenshen2019,
  title={Emoji-powered representation learning for cross-lingual sentiment classification},
  author={Zhenpeng Chen and Sheng Shen and Ziniu Hu and Xuan Lu and Qiaozhu Mei and Xuanzhe Liu},
  booktitle={Proceedings of the 2019 World Wide Web Conference on World Wide Web, {WWW} 2019},
  year={2019}
}
```

 

