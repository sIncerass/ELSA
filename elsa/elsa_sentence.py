from __future__ import print_function
import numpy as np
from keras.layers import *
from keras.models import Model, Sequential
from keras.layers.merge import concatenate
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation
from keras.regularizers import L1L2 
from attlayer import AttentionWeightedAverage
from avglayer import MaskAverage
from global_variables import NB_TOKENS, NB_EMOJI_CLASSES
from copy import deepcopy
from os.path import exists
import h5py
import uuid, os
from keras.optimizers import Adam
from finetuning import (sampling_generator, finetuning_callbacks)
import yaml

def elsa_architecture(nb_classes, nb_tokens, maxlen, feature_output=False, embed_dropout_rate=0, final_dropout_rate=0, embed_dim=300,
                    embed_l2=1E-6, return_attention=False, load_embedding=False, pre_embedding=None, high=False, LSTM_hidden=512, LSTM_drop=0.5):
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
        embed = Embedding(input_dim=nb_tokens, output_dim=embed_dim, mask_zero=True,input_length=maxlen,embeddings_regularizer=embed_reg,
                          name='embedding')
    else:
        embed = Embedding(input_dim=nb_tokens, output_dim=embed_dim, mask_zero=True,input_length=maxlen, weights=[pre_embedding],
                          embeddings_regularizer=embed_reg,trainable=True, name='embedding')
    if high:
        x = NonMasking()(embed(model_input))
    else:
        x = embed(model_input)
    x = Activation('tanh')(x)

    # entire embedding channels are dropped out instead of the
    # normal Keras embedding dropout, which drops all channels for entire words
    # many of the datasets contain so few words that losing one or more words can alter the emotions completely
    if embed_dropout_rate != 0:
        embed_drop = SpatialDropout1D(embed_dropout_rate, name='embed_drop')
        x = embed_drop(x)

    # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
    # ordering of the way the merge is done is important for consistency with the pretrained model
    lstm_0_output = Bidirectional(LSTM(LSTM_hidden, return_sequences=True, dropout=LSTM_drop), name="bi_lstm_0" )(x)
    lstm_1_output = Bidirectional(LSTM(LSTM_hidden, return_sequences=True, dropout=LSTM_drop), name="bi_lstm_1" )(lstm_0_output)
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
        if final_dropout_rate != 0:
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


if __name__ == '__main__':
    config = yaml.load( open('elsa_train.yaml') )
    os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU_ID']
    cur_lan = config['cur_lan']
    maxlen = config['maxlen']
    batch_size = config['batch_size']
    lr = config['lr']
    epoch_size = config['epoch_size']
    nb_epochs = config['nb_epochs']
    patience = config['patience']
    checkpoint_weight_path = config['checkpoint_weight_path']
    loss = config['loss']
    optim = config['optim']
    vocab_path = config['vocab_path']

    steps = int(epoch_size/batch_size)

    word_vec = np.load(vocab_path + "%s_wv.npy" % cur_lan)
    input_vec, input_label = np.load(vocab_path + "%s_input.npy" % cur_lan), np.load(vocab_path + "%s_labels.npy" % cur_lan)
    nb_tokens, input_len = len(word_vec), len(input_label)

    #please modify the checkpoint_weight_path
    checkpoint_weight_path = '/storage1/user/ss/tmoji_ori/weight/tmoji-lstm-checkpoint-%s-h-1.hdf5' % cur_lan

    idx_shuffle = list(range(input_len))
    np.random.shuffle(idx_shuffle)
    idx_train, idx_val, idx_test = idx_shuffle[ :int(input_len*0.7) ], idx_shuffle[int(input_len*0.7):int(input_len*0.9)], idx_shuffle[int(input_len*0.9):]

    (X_train, y_train) = (input_vec[idx_train], input_label[idx_train])
    (X_val, y_val) = (input_vec[idx_val], input_label[idx_val])
    (X_test, y_test) = (input_vec[idx_test], input_label[idx_test])
    LSTM_hidden = config['LSTM_hidden']
    LSTM_drop = config['LSTM_dropout']
    final_dropout_rate = config['final_dropout_rate']
    embed_dropout_rate = config['embed_dropout_rate']
    high = config['high']
    load_embedding = config['load_embedding']
    embed_dim = config['embed_dim']
    model = elsa_architecture(nb_classes=64, nb_tokens=nb_tokens, maxlen=maxlen, final_dropout_rate=final_dropout_rate, embed_dropout_rate=embed_dropout_rate, 
                            load_embedding=True, pre_embedding=word_vec, high=high, embed_dim=embed_dim)
    model.summary()
    if optim == 'adam':
        adam = Adam(clipnorm=1, lr=lr)
        model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    elif optim == 'rmsprop':
        model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])
    callbacks = finetuning_callbacks(checkpoint_weight_path, patience, verbose=1)
    for i in range(2):
        train_gen = sampling_generator(X_train, y_train, batch_size, upsample=False, epoch_size=epoch_size)
        model.fit_generator(train_gen, steps_per_epoch=steps, epochs=nb_epochs,validation_data=(X_val, y_val),validation_steps=steps, callbacks=callbacks, verbose=True)
    _, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print(acc)