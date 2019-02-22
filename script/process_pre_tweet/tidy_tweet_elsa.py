import codecs
import numpy as np
import re, json, sys
reload(sys)
sys.setdefaultencoding('utf-8')
from collections import defaultdict, OrderedDict

#configure
cur_lan = "en" # "de" or "fr" or "jp"
input_file_name = "tmoji_tokens_%s" % cur_lan # precessed tweets after running the scripts in process_raw
vocab_path = "/storage1/user/ss/vocab/" # path to store the numpy version of processed training and testing tweets
pre_vocab_file = vocab_path + "en_vocab.json"
top_num = 64
#end configure

emoji = defaultdict(lambda: 0, {})
with open(input_file_name, "r") as stream:
    for line in stream:
        word = line.split('\t')[0]
        try:
            emoji[word] += file_name[f]
        except KeyError:
            emoji[word] = file_name[f]
    stream.close()

sorted_emoji = sorted(emoji.items(), key=lambda x: x[1], reverse=True)
total = [0, 0]
wanted_emoji = []
with open(vocab_path + "emoji_%s" % cur_lan , "w") as f:
    for i, emoji in enumerate(sorted_emoji):
        f.write("%s\t%d\n" % (emoji[0], emoji[1]))
        if i < top_num:
            total[1] += emoji[1]
            wanted_emoji.append(emoji[0])
        total[0] += emoji[1]
print(wanted_emoji)
open("emoji_%s_top%s.json" % (cur_lan, top_num), "w").write( json.dumps( wanted_emoji  )  )
with open(vocab_path + "want_tmoji_%s" % cur_lan, "w") as f:
    for i, emoji in enumerate(sorted_emoji):
        if i < top_num:
            f.write("%s\t%d\n" % (emoji[0], emoji[1]))
open("%semoji_%s_top%s.json" % (vocab_path, cur_lan, top_num), "w").write( json.dumps( wanted_emoji  )  )

wanted_emoji = json.loads( open("%semoji_%s_top%s.json" % (vocab_path, cur_lan, top_num),"r").read() )
emoji_filter = [x.decode('utf-8') for x in wanted_emoji]
tidy_data = []
for f in file_name:
    with open(f, "r") as stream:
        for line in stream:
            data = line.strip().split("\t")
            if data[0] in emoji_filter:
                if len(json.loads(data[1])) > 2:
                    tidy_data.append((data[0], json.loads(data[1])))
print(wanted_emoji, emoji_filter, len(wanted_emoji), tidy_data[0])

import math
def calculate_batchsize_maxlen(texts):
    """ Calculates the maximum length in the provided texts and a suitable
        batch size. Rounds up maxlen to the nearest multiple of ten.
    # Arguments:
        texts: List of inputs.
    # Returns:
        Batch size,
        max length
    """
    def roundup(x):
        return int(math.ceil(x / 10.0)) * 10
    # Calculate max length of sequences considered
    # Adjust batch_size accordingly to prevent GPU overflow
    lengths = [len(t) for _, t in texts]
    maxlen = roundup(np.percentile(lengths, 80.0))
    batch_size = 250 if maxlen <= 100 else 50
    print("mean: ", np.mean(lengths), "median: ", np.median(lengths), len(lengths), "avg: ", np.average(lengths))
    print("batch_size: ", batch_size, "maxlen:", maxlen)
    return batch_size, maxlen
batch_size, maxlen = calculate_batchsize_maxlen(tidy_data)

vocab_index = json.loads(open(pre_vocab_file, "r").read())
n_sentences = len(tidy_data)
fixed_length = maxlen
def find_tokens(words):
    assert len(words) > 0
    tokens = []
    for w in words:
        try:
            tokens.append(vocab_index[w])
        except KeyError:
            tokens.append(1)
    return tokens
infos = []
tokens = np.zeros((n_sentences, fixed_length), dtype='uint32')
next_insert = 0
n_ignored_unknowns = 0
for s_info, s_words in tidy_data:
    s_tokens = find_tokens(s_words)
    if len(s_tokens) > fixed_length:
        s_tokens = s_tokens[:fixed_length]
    tokens[next_insert,:len(s_tokens)] = s_tokens
    tmp_info = np.zeros(64)
    tmp_info[ wanted_emoji.index(s_info) ] = 1
    infos.append(tmp_info)
    next_insert += 1
del tidy_data

#balance the input
balance_emoji = {x:[] for x in range(64)}
print(len(infos), len(tokens))
for i, info in enumerate(infos):
    k = np.argmax(info)
    balance_emoji[k].append(i)
train, val, test = [], [], []
for item in balance_emoji:
    line = balance_emoji[item]
    np.random.shuffle(line)
    length = len(line)
    train += line[:int(length*0.7)]
    val += line[int(length*0.7):int(length*0.9)]
    test += line[int(length*0.9):]
np.random.shuffle(train), np.random.shuffle(test), np.random.shuffle(val)
filter_token = []
filter_info = []
print(train[:5], test[:5], val[:5])
total = train + test + val
for index in total:
    filter_token.append(tokens[index])
    filter_info.append(infos[index])
print(len(filter_info), len(filter_token))
# finally processed info and label as emoji tweets
np.save(vocab_path + "%s_labels" % cur_lan, filter_info)
np.save(vocab_path + "%s_input" % cur_lan, filter_token)
