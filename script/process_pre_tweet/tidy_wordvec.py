# -*- coding: utf-8 -*-
import gensim
import json, time

# given that the computer might not be able to load the whole document, 
# we split it and train them recursively
input_model, output_model = 'to_tmp_1', 'to_tmp_0'
output_file = "tweet_to_w.txt"
input_file =  "tmoji_en_0"
total_num = 20496791 # number of tweets
split_num, split_id = 5, 0
model = gensim.models.Word2Vec.load(input_model)
model.model_trimmed_post_training = False
sentence = []
start = total_num // split_num * split_id
end = total_num // split_num * ( split_id + 1)
start_time = time.clock()
with open(input_file, "r") as stream:
	for i, line in enumerate(stream):
		if i <= start:
			continue
		if i > end:
			break
		sentence.append(json.loads(line))
	stream.close()
print("read", time.clock()-start_time)
model.build_vocab(sentence, update=True)
model.train(sentence, total_examples=model.corpus_count,epochs=model.iter)
print("train", time.clock()-start_time)
model.save(output_model)
model.wv.save_word2vec_format(output_file, binary=False)

