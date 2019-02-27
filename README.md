# ELSA

ELSA is an emoji-powered representation learning framework for cross-lingual sentiment classification. 

The workflow of ELSA consists of the following phases:

1. It uses large-scale Tweets to learn word embeddings through Word2Vec of both the source and the target languages in an unsupervised way. 

2. In a distant-supervised way, it uses emojis as complementary sentiment labels to transform the word embeddings into a higher-level sentence representation that encodes rich sentiment information via an emoji-prediction task through an attention-based stacked bi-directional LSTM model. This step is also conducted separately for both the source and the target languages. 

3. It translates the labeled English data into the target language through an off-the-shelf machine translation system, represent the pseudo parallel texts with the pre-trained language-specific models, and use an attention model to train the final sentiment classifier.

You can see the WWW 2019 (know as The Web Conference) paper “**Emoji-Powered Representation Learning for Cross-Lingual Sentiment Classification**” for more details.

## Overview

- dataset/ 
  contains the raw and processed data used for evaluating our approach. It contains two subfolders: 
  - Amazon review/ 
    contains the pre-processed Amazon review dataset created by [Prettenhofer and Stein](http://www.aclweb.org/anthology/P10-1114). Aside from the given parallel texts of the test data (i.e., the Japanese, French and German reviews), we translate English reviews into Japanese, French, and German through [Google Translate](https://translate.google.com). Each line of these included files is composed of `sentiment label \t english version \t other language version`, please use the json files to parse each review (e.g., `./en_de/de/books_train_review.tsv`, they have already been processed into list of words)
- scripts/ 
  contains the scripts for pre-processing Tweets and training word embeddings.
  - process_raw_tweet/ contains the scripts of tokenizing and extracting emojis from Tweets. You can modify `tweet_token.py` with `input_file, output_file, emoji_file` field for different tasks. Set `JAPAN=True/False` in the `word_generator.py` file for pre-processing Tweets when dealing with different languages.
  - process_pre_tweets/ contains the vocabulary file that can be generated through scripts in `process_raw_tweet/ `. Please change the filename and path name in each script and run `tidy_wordvec.py, tidy_vocab.py, tidy_tweet_elsa.py` in sequence to learn the word embeddings and tidy the Tweets into the format of model input. 
- elsa/ 
  contains the core scripts (`.py` files) and configuration files (`.yaml` files) of ELSA and you can turn to the detailed instructions in the following Setup to run ELSA. The core scripts include:
  - elsa_sentence.py : train a new version of sentence representation model of ELSA from pre-processed Tweets.
  - test_elsa_sentence.py : generate sentence representations for Amazon review dataset in respective language setting.
  - elsa_doc.py : train the final sentiment classifier of ELSA.
- pretrained_model/ 
  contains contains  the pre_trained models in this study, including the representation models (Download link: https://drive.google.com/drive/folders/13dQdhLl3ZZogM3B1AV_xpI8dGe_wPY3K?usp=sharing) and the final sentiment classifier (e.g., en_de/books_weights_t_att.hdf5 files).

## Setup

1. We assume that you're using Python 3.6 with pip installed. As a backend you need to install either Theano (version 0.9+) or Tensorflow (version 1.3+). To run the code, you need the following dependencies:

- [Keras](https://github.com/keras-team/keras) (above 2.0.0)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [text-unidecode](https://github.com/kmike/text-unidecode)
- [Mecab](http://taku910.github.io/mecab/) tokenize Japanese.
- [yaml](https://github.com/yaml)

You can use the python package manager of your choice (*pip/conda*) to install the dependencies.
The code is tested on the *Linux* operating system. 

2. To reproduce our main results, for the representation learning phase of ELSA, you can change the configuration file `elsa_test.yaml` by setting the

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

3. After detailed pre-processing of Tweets and dataset decribed above, in order to train a new representation model of ELSA, you can run the scripts in the elsa/ directory and change the `elsa_train.yaml` as you please. 

   Furthermore, to train a new final sentiment classifier, after obtaining the sentence representation for each sentence in the docuement, you can modify the `mode: 'train'`in `elsa_doc.yaml` file and fine-tune your own model accordingly.

## Dataset

We sadly cannot release our large-scale dataset of Tweets used to train representation learning models due to licensing restrictions.

We upload all the benchmark datasets to this repository for convenience. As they were not collected and released by us, we do not claim any rights on them. If you use any of these datasets, please make sure you fulfill the licenses that they were released with and consider citing the original authors.

### Visualization

![Alt text](https://github.com/sIncerass/ELSA/raw/master/pics/neg_e.png)

### Citation

Please consider citing the following paper when using our code or pretrained models for your application.

```
@inproceedings{chenshen2019,
  title={Emoji-powered representation learning for cross-lingual sentiment classification},
  author={Zhenpeng Chen and Sheng Shen and Ziniu Hu and Xuan Lu and Qiaozhu Mei and Xuanzhe Liu},
  booktitle={Proceedings of the 2019 World Wide Web Conference},
  year={2019}
}
```
