import json
import gensim
import numpy as np

# tidy the vocabulary for each language
vocab = json.loads(open("jp_vocab.json", "r").read())
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=False)
sorted_vocab = [x for x, _ in sorted_vocab]
# the pre-defined placeholder
print(sorted_vocab[:10])
tmp_vocab = sorted_vocab[:10]

vocab_path = "/storage1/user/ss/tmoji_ori/data/vocab/" # please modify this field to store the according vocab, wordvec.npy file
model_vec = "tweet_en_wv.txt" # the word embedding file 
out_dict = "en_vocab.json" # target vocabulary file
out_vec = "vocab/en_wv" # numpy verison of pretrained word embedding
with open(model_vec, "r") as stream:
    for i, line in enumerate(stream):
        if i > 0:
            data = line.strip().split(' ')
            if data[0].decode('utf-8') not in tmp_vocab:
                tmp_vocab.append(data[0].decode('utf-8'))
print(tmp_vocab[0])
vocab_index = {x:i for i, x in enumerate(tmp_vocab)}
open(vocab_path+out_dict, "w").write(json.dumps(vocab_index))

model = gensim.models.KeyedVectors.load_word2vec_format(model_vec, binary=False)
word2vec, fail = [], 0
for word in tmp_vocab:
    try:
        word2vec.append( model[word] )
    except:
        print(word, fail)
        fail += 1
        word2vec.append( np.random.uniform(-1, 1, 300) )
np.save(vocab_path+out_vec, word2vec)

