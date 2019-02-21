# -*- coding: utf-8 -*-
from __future__ import print_function, division
from keras.layers import *
from keras.models import Model, Sequential
from keras.layers.merge import concatenate
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation
from keras.regularizers import L1L2
from attlayer import AttentionWeightedAverage
import json, os, csv
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from keras.models import load_model
import yaml

def elsa_architecture(nb_classes, nb_tokens, maxlen, feature_output=False, embed_dropout_rate=0, final_dropout_rate=0, embed_l2=1E-6, 
                    return_attention=False, load_embedding=False, pre_embedding=None, high=False, test=False, LSTM_drop=0.5, LSTM_hidden=512):
    """
    Returns the DeepMoji architecture uninitialized and
    without using the pretrained model weights.
    # Arguments:
        nb_classes: Number of classes in the dataset.
        nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
        maxlen: Maximum length of a token.
        feature_output: If True the model returns the penultimate
                        feature vector rather than Softmax probabilities
                        (defaults to False).
        embed_dropout_rate: Dropout rate for the embedding layer.
        final_dropout_rate: Dropout rate for the final Softmax layer.
        embed_l2: L2 regularization for the embedding layerl.
        high: use or not the highway network
    # Returns:
        Model with the given parameters.
    """
    class NonMasking(Layer):   
        def __init__(self, **kwargs):   
            self.supports_masking = True  
            super(NonMasking, self).__init__(**kwargs)   

        def build(self, input_shape):   
            input_shape = input_shape   

        def compute_mask(self, input, input_mask=None):   
            # do not pass the mask to the next layers   
            return None   

        def call(self, x, mask=None):   
            return x   

        def get_output_shape_for(self, input_shape):   
            return input_shape 
    # define embedding layer that turns word tokens into vectors
    # an activation function is used to bound the values of the embedding
    model_input = Input(shape=(maxlen,), dtype='int32')
    embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
    if not load_embedding and pre_embedding is None:
        embed = Embedding(input_dim=nb_tokens,output_dim=300, mask_zero=True,input_length=maxlen,embeddings_regularizer=embed_reg,
                          name='embedding')
    else:
        embed = Embedding(input_dim=nb_tokens, output_dim=300,mask_zero=True,input_length=maxlen, weights=[pre_embedding],
                          embeddings_regularizer=embed_reg,trainable=True, name='embedding')
    if high:
        x = NonMasking()(embed(model_input))
    else:
        x = embed(model_input)
    x = Activation('tanh')(x)

    # entire embedding channels are dropped out instead of the
    # normal Keras embedding dropout, which drops all channels for entire words
    # many of the datasets contain so few words that losing one or more words can alter the emotions completely
    if not test and embed_dropout_rate != 0:
        embed_drop = SpatialDropout1D(embed_dropout_rate, name='embed_drop')
        x = embed_drop(x)

    # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
    # ordering of the way the merge is done is important for consistency with the pretrained model
    lstm_0_output = Bidirectional(LSTM(LSTM_hidden, return_sequences=True, dropout=0.0 if test else LSTM_drop), name="bi_lstm_0" )(x)
    lstm_1_output = Bidirectional(LSTM(LSTM_hidden, return_sequences=True, dropout=0.0 if test else LSTM_drop), name="bi_lstm_1" )(lstm_0_output)
    x = concatenate([lstm_1_output, lstm_0_output, x])
    if high:
        x = TimeDistributed(Highway(activation='tanh', name="high"))(x)
    # if return_attention is True in AttentionWeightedAverage, an additional tensor
    # representing the weight at each timestep is returned
    weights = None
    x = AttentionWeightedAverage(name='attlayer', return_attention=return_attention)(x)
    #x = MaskAverage(name='attlayer', return_attention=return_attention)(x)
    if return_attention:
        x, weights = x

    if not feature_output:
        # output class probabilities
        if not test and final_dropout_rate != 0:
            x = Dropout(final_dropout_rate)(x)

        if nb_classes > 2:
            outputs = [Dense(nb_classes, activation='softmax', name='softmax')(x)]
        else:
            outputs = [Dense(1, activation='sigmoid', name='softmax')(x)]
    else:
        # output penultimate feature vector
        outputs = [x]

    if return_attention:
        # add the attention weights to the outputs if required
        outputs.append(weights)

    return Model(inputs=[model_input], outputs=outputs)

def emoji_test(model, vocab_path, cur_lan):
    input_vec, input_label = np.load(vocab_path + "%s_input.npy" % cur_lan), np.load(vocab_path + "%s_labels.npy" % cur_lan)
    nb_tokens, input_len = len(word_vec), len(input_label)
    (X_test, y_true) = (input_vec[int(input_len*0.9):], input_label[int(input_len*0.9):])
    y_test = model.predict(X_test)
    def top_n_accuracy(preds, truths, n):
        best_n = np.argsort(preds, axis=1)[:,-n:]
        ts = np.argmax(truths, axis=1)
        print(best_n)
        successes = 0
        for i in range(ts.shape[0]):
            if ts[i] in best_n[i,:]:
                successes += 1
        return float(successes)/ts.shape[0]
    topk = top_n_accuracy(y_true, y_test, n=5)
    predicted_categories = [np.argmax(x) for x in y_test]
    expected_categories = [np.argmax(x) for x in y_true]
    print("topk: ", topk)
    print(classification_report(expected_categories, predicted_categories))

if __name__ == '__main__':
    config = yaml.load( open('elsa_test.yaml') )
    os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU_ID']
    vocab_path = config['vocab_path']
    weight_path = config['checkpoint_weight_path']
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    cur_lan = config[ 'cur_lan' ]
    cur_test = "en_%s/" % cur_lan

    vocab_index = json.loads(open(vocab_path + "%s_vocab.json" % cur_lan, "r").read())
    word_vec = np.load(vocab_path + "%s_wv.npy" % cur_lan)
    nb_tokens, fixed_length = len(word_vec), config['maxlen']
    
    model = elsa_architecture(nb_classes=config['nb_classes'], nb_tokens=nb_tokens, maxlen=fixed_length, feature_output=False, return_attention=False, test=config['test'])
    model.load_weights(weight_path, by_name=True)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('attlayer').output)

    intermediate_layer_model.summary()
    def find_tokens(words):
        assert len(words) > 0
        tokens = []
        for w in words:
            try:
                tokens.append(vocab_index[w])
            except KeyError:
                tokens.append(1)
        return tokens
    batch_size = config[ 'batch_size' ]
    cur_en_or_ot = config[ 'cur_en_or_ot' ]
    if cur_en_or_ot:
        cur_config_lan = 'en'
    else:
        cur_config_lan = cur_lan
    embed_files = os.listdir(input_dir+cur_test+'en')

    for embed_chose in range(2):
        embed_file = "en" if embed_chose else cur_test[-3:-1]
        for cur_file in embed_files:
            doc_embedding, doc_file = [], []
            next_insert = 0
            print(cur_config_lan, cur_file)
            cur_ = open(input_dir+cur_test+embed_file+"/"+cur_file, "r")
            for data in cur_:
                line = data.strip().split('\t')
                if cur_config_lan != "en":
                    data = line[2] if embed_chose else line[1]
                else:
                    data = line[1] if embed_chose else line[2]
                data = json.loads(data)
                tokens = np.zeros((len(data), fixed_length), dtype='uint32')
                next_insert = 0
                for s_words in data:
                    s_tokens = find_tokens(s_words)
                    if len(s_tokens) > fixed_length:
                        s_tokens = s_tokens[:fixed_length]
                    tokens[next_insert,:len(s_tokens)] = s_tokens
                    next_insert += 1
                doc_file.append(tokens)
            cur_.close()
            for embed_sen in doc_file:
                encoding = intermediate_layer_model.predict(embed_sen)
                doc_embedding.append(encoding)
            np.save(output_dir+cur_test+cur_config_lan+"/"+embed_file+"_"+cur_file.replace(".tsv", "_embed.npz"), doc_embedding)
