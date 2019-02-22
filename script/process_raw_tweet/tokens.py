# -*- coding: utf-8 -*-
'''
Splits up a Unicode string into a list of tokens.
Recognises:
- Abbreviations
- URLs
- Emails
- #hashtags
- @mentions
- emojis
- emoticons (limited support)

Multiple consecutive symbols are also treated as a single token.
'''

import re

# Basic patterns.
RE_NUM = u'[0-9]+'
RE_WORD = u'[a-zA-Z]+'
RE_WHITESPACE = u'\s+'
RE_ANY = u'.'

# Combined words such as 'red-haired' or 'CUSTOM_TOKEN'
RE_COMB = u'[a-zA-Z]+[-_][a-zA-Z]+'

# English-specific patterns
RE_CONTRACTIONS = RE_WORD + u'\'' + RE_WORD

TITLES = [
    u'Mr\.',
    u'Ms\.',
    u'Mrs\.',
    u'Dr\.',
    u'Prof\.',
    ]
# Ensure case insensitivity
RE_TITLES = u'|'.join([u'(?i)' + t for t in TITLES])

# Symbols have to be created as separate patterns in order to match consecutive
# identical symbols.
SYMBOLS = u'()<!?.,/\'\"-_=\\§|´ˇ°[]<>{}~$^&*;:%+\xa3€`'
RE_SYMBOL = u'|'.join([re.escape(s) + u'+' for s in SYMBOLS])

# Hash symbols and at symbols have to be defined separately in order to not
# clash with hashtags and mentions if there are multiple - i.e.
# ##hello -> ['#', '#hello'] instead of ['##', 'hello']
SPECIAL_SYMBOLS = u'|#+(?=#[a-zA-Z0-9_]+)|@+(?=@[a-zA-Z0-9_]+)|#+|@+'
RE_SYMBOL += SPECIAL_SYMBOLS

RE_ABBREVIATIONS = u'\b(?<!\.)(?:[A-Za-z]\.){2,}'

# Twitter-specific patterns
RE_HASHTAG = u'#[a-zA-Z0-9_]+'
RE_MENTION = u'@[a-zA-Z0-9_]+'

RE_URL = u'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
RE_EMAIL = u'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b'

# Emoticons and emojis
RE_HEART = u'(?:<+/?3+)+'
EMOTICONS_START = [
    u'>:',
    u':',
    u'=',
    u';',
    ]
EMOTICONS_MID = [
    u'-',
    u',',
    u'^',
    u'\'',
    u'\"',
    ]
EMOTICONS_END = [
    u'D',
    u'd',
    u'p',
    u'P',
    u'v',
    u')',
    u'o',
    u'O',
    u'\(',
    u'3',
    u'/',
    u'|',
    u'\\',
    ]
EMOTICONS_EXTRA = [
    u'-_-',
    u'x_x',
    u'^_^',
    u'o.o',
    u'o_o',
    u'\(:',
    u'):',
    u');',
    u'\(;',
    ]

RE_EMOTICON = u'|'.join([re.escape(s) for s in EMOTICONS_EXTRA])
for s in EMOTICONS_START:
    for m in EMOTICONS_MID:
        for e in EMOTICONS_END:
            RE_EMOTICON += '|{0}{1}?{2}+'.format(re.escape(s), re.escape(m), re.escape(e))

# requires ucs4 in python2.7 or python3+
# RE_EMOJI = r"""[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]"""
# safe for all python
RE_EMOJI = u"""\ud83c[\udf00-\udfff]|\ud83d[\udc00-\ude4f\ude80-\udeff]|\ud83e[\udd10-\uddff]|[\u2600-\u26FF\u2700-\u27BF]"""

# List of matched token patterns, ordered from most specific to least specific.
TOKENS = [
    RE_URL,
    RE_EMAIL,
    RE_COMB,
    RE_HASHTAG,
    RE_MENTION,
    RE_HEART,
    RE_EMOTICON,
    RE_CONTRACTIONS,
    RE_TITLES,
    RE_ABBREVIATIONS,
    RE_NUM,
    RE_WORD,
    RE_SYMBOL,
    RE_EMOJI,
    RE_ANY
    ]

# List of ignored token patterns
IGNORED = [
    RE_WHITESPACE
    ]

# Final pattern
RE_PATTERN = re.compile(u'|'.join(IGNORED) + u'|(' + u'|'.join(TOKENS) + u')',
                        re.UNICODE)


def tokenize(text):
    '''Splits given input string into a list of tokens.

    # Arguments:
        text: Input string to be tokenized.

    # Returns:
        List of strings (tokens).
    '''
    result = RE_PATTERN.findall(text)

    # Remove empty strings
    result = [t for t in result if t.strip()]
    return result
