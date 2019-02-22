import codecs, json, time
from create_vocab import VocabBuilder
from word_generator import *

substitute = [('CUSTOM_URL', ' ;; '), ('CUSTOM_UNKNOWN', ' [[ '), ('CUSTOM_NUMBER', ' ]] '), ('CUSTOM_MASK', ' {{ '), ('CUSTOM_BREAK', ' }} '), ('CUSTOM_AT', ' ,, ')]

emoji_unicodes = json.loads(open("emoji_unicode", "r").read())
emoji_unicode = emoji_unicodes.keys()
def check_ascii(word):
    try:
        word.decode('ascii')
        return True
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False

#please modify these three field for preprocessing the tweets
input_file = "elsa_fr_raw"
output_file = "elsa_fr_processed"
emoji_file = "elsa_fr_top_emoji"

emoji_frequency = {x:0 for x in emoji_unicode}
cur_tokens = []
start_time = time.clock()

total_file = codecs.open(output_file, "w")
# with codecs.open('../twitter_data/Twitter_update_filtered.tsv', 'rU', 'utf-8') as stream:
with codecs.open(input_file+'.tsv', 'rU') as stream:
    wg = TweetWordGenerator(stream)
    vb = VocabBuilder(wg)

    numbers, emoji_number, emoji_senten = 0, 0, 0
    for i, (tokens, info) in enumerate(wg):
        numbers += 1
        emoji_en_senten = 0
        for item in tokens:
            if item in emoji_unicode:
                emoji_number += 1
                emoji_en_senten = 1
                emoji_frequency[item] += 1
        if i % 10000 == 0:
            print(i, tokens, float(i), time.clock()-start_time, emoji_number)
        emoji_senten += emoji_en_senten
        total_file.write("%s\n" %  json.dumps(tokens))
    print(numbers, emoji_senten, emoji_number, float(emoji_senten)/numbers)
    
    sorted_emoji = sorted(emoji_frequency.items(), key=lambda x: x[1], reverse=True)
    with codecs.open(emoji_file, "w", 'utf-8') as f:
        for emoji in sorted_emoji:
            if emoji[1] > 0:
                f.write("%s\t%d\n" % (emoji[0], emoji[1]))

