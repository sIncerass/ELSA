#label file
import pandas as pd
import json, os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import *
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from attlayer import AttentionWeightedAverage
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import yaml

def elsa_doc_model(hidden_dim = 64, dropout = 0.5, mode = 'train'):
    I_en = Input(shape=(nb_maxlen[0], nb_feature[1]), dtype='float32')
    en_out = AttentionWeightedAverage()(I_en)
    I_ot = Input(shape=(nb_maxlen[1], nb_feature[0]), dtype='float32')
    jp_out = AttentionWeightedAverage()(I_ot)
    O_to = concatenate([jp_out, en_out])
    O_to = Dense(hidden_dim, activation='selu')(O_to)
    if mode == 'train':
        O_to = Dropout(dropout)(O_to)
    O_out = Dense(1, activation='sigmoid', name='softmax')(O_to)
    model = Model(inputs=[I_ot, I_en], outputs=O_out)
    return model

if __name__ == '__main__':
    config = yaml.load( open('elsa_doc.yaml') )
    os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU_ID']
    cur_cat = config['cur_cat']
    cur_test = config[ 'cur_test' ]
    nb_feature = config[ 'nb_feature' ]
    nb_maxlen = config[ 'nb_maxlen' ]
    label_path = config[ 'label_path' ]
    embed_path = config[ 'embed_path' ]
    labes = {"en_test_review":[], "en_train_review":[], cur_test[-3:-1]+"_test_review":[], cur_test[-3:-1]+"_train_review":[]}
    tags = ["en_test_review", "en_train_review", cur_test[-3:-1]+"_test_review", cur_test[-3:-1]+"_train_review"]
    filename = [label_path+cur_test+"en/"+cur_cat[1:]+"_test_review.tsv", label_path+cur_test+"en/"+cur_cat[1:]+"_train_review.tsv", 
                label_path+cur_test+cur_test[-3:]+cur_cat[1:]+"_test_review.tsv", label_path+cur_test+cur_test[-3:]+cur_cat[1:]+"_train_review.tsv"]
    for i, file in enumerate(filename):
        data = open(file, "r")
        for line in data:
            tmp_data = line.strip().split("\t")
            rating = int(tmp_data[0])
            if rating > 3:
                labes[tags[i]].append(1)
            else:
                labes[tags[i]].append(0)
        data.close()

    # tidy elsa_embedding
    elsa_embedding = {x:[np.array([]), np.array([])] for x in tags}
    def roundup(x):
        import math
        return int(math.ceil(x / 10.0)) * 10
    for tag in tags:
        tmp_tag = tag[:2] + cur_cat + tag[2:]
        vec = np.load(embed_path+cur_test+ cur_test[-3:]+tmp_tag+"_embed.npz.npy")
        vec = sequence.pad_sequences(vec, dtype=np.float32, maxlen=nb_maxlen[0])
        elsa_embedding[tag][0] = vec   
    for tag in tags:
        tmp_tag = tag[:2] + cur_cat + tag[2:]
        vec = np.load(embed_path+cur_test+"en/"+tmp_tag+"_embed.npz.npy")
        vec = sequence.pad_sequences(vec, dtype=np.float32, maxlen=nb_maxlen[1])
        elsa_embedding[tag][1] = np.array(vec)
        print(vec.shape, vec[0].shape)

    # train elsa_doc model
    weigh_path = config['weight_path']
    batch_size = config['batch_size']
    epochs = config['epochs']
    hidden_dim = config['hidden_dim']
    dropout = config['dropout']
    mode = config['mode']
    elsa_doc = elsa_doc_model( hidden_dim=hidden_dim, dropout=dropout, mode=mode )
    elsa_doc.summary()
    if mode == 'train':
        ck = ModelCheckpoint(filepath=weigh_path, verbose=0, save_best_only=True, monitor='val_acc')
        elsa_doc.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        test_chose = config['train_chose']
        tmp_x = elsa_embedding['en_train_review'] if test_chose else elsa_embedding['en_test_review']
        tmp_y = labes['en_train_review'] if test_chose else labes['en_test_review']
        test_x = elsa_embedding['en_test_review'] if test_chose else elsa_embedding['en_train_review']
        test_y = labes['en_test_review'] if test_chose else labes['en_train_review']
        elsa_doc.fit([tmp_x[0], tmp_x[1]], tmp_y, batch_size=batch_size, epochs=epochs, validation_data=([test_x[0], test_x[1]], test_y), verbose=True, callbacks=[ck])
    else:
        pretrained_path = config['pretrain_path'] + cur_test + cur_cat[1:] + "_weights_t_att.hdf5"
        elsa_doc.load_weights(filepath=pretrained_path)
        test_x = elsa_embedding[cur_test[-3:-1:]+'_test_review']
        test_y = labes[cur_test[-3:-1:]+'_test_review']
        predict_total = elsa_doc.predict([test_x[0], test_x[1]])
        predict_total = [int(x > 0.5) for x in predict_total]
        acc = accuracy_score(predict_total, test_y)
        print("%s %s Test Accuracy: %s" %  (cur_test[:-1], cur_cat[1:], acc) )
