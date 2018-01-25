''' Preporcessing steps:
1. lowercasing 
2. Digit -> DDD 
3. URLs -> httpAddress 
4. @username -> userID 
5. Remove special characters, keep ; . ! ? 
6. normalize elongation 
7. tokenization using tweetNLP'''

# =================
# ==> Libraries <==
# =================
import re, os
import string
import sys
# from twisted.python.test.test_inotify import libc

import twokenize
import csv
from collections import defaultdict
from os.path import basename
import ntpath
import codecs
import unicodedata

# =============
# ==> Paths <==
# =============
prccd_folder = "../data"  # no backslashes in front of special characters like spaces
prccd_folder = os.path.expanduser(prccd_folder)


# =================
# ==> Functions <==
# =================
# tweet = "HIII! 1000 http://wwww.google.com @ALT :)"

def process(lst):
    prccd_item_list = []
    for tweet in lst:
        # print "[original]", tweet
        # print(tweet)

        # 0. Removing special Characters
        # punc = '\''
        # trans = string.maketrans(punc, ' '*len(punc))
        # tweet = tweet.translate(trans)#quan them vao

        # Normalizing utf8 formatting
        tweet = tweet.decode("unicode-escape").encode("utf8").decode("utf8")
        # tweet = tweet.encode("utf-8")
        tweet = tweet.encode("ascii", "ignore")
        tweet = tweet.strip(' \t\n\r')

        # 1. Lowercase
        tweet = tweet.lower()
        # print "[lowercase]", tweet

        # Word-Level
        tweet = re.sub(' +', ' ', tweet)  # replace multiple spaces with a single space

        # 2. Normalizing digits
        tweet_words = tweet.strip('\r').split(' ')
        for word in [word for word in tweet_words if word.isdigit()]:
            tweet = tweet.replace(word, "D" * len(word))
        #		print "[digits]", tweet
        #  3. Normalizing URLs
        tweet_words = tweet.strip('\r').split(' ')
        for word in [word for word in tweet_words if '/' in word or '.' in word and len(word) > 3]:
            tweet = tweet.replace(word, "httpAddress")  # print "[URLs]", tweet
        #  4. Normalizing username
        tweet_words = tweet.strip('\r').split(' ')
        for word in [word for word in tweet_words if word[0] == '@' and len(word) > 1]:
            tweet = tweet.replace(word, "usrId")
            # print "[usrename]", tweet

        # 5. Removing special Characters
        punc = '@$%^&*()_+-={}[]:"|\'\~`<>/,'
        trans = string.maketrans(punc, ' ' * len(punc))
        tweet = tweet.translate(trans)
        # print "[punc]", tweet

        # 6. Normalizing +2 elongated char
        tweet = re.sub(r"(.)\1\1+", r'\1\1', tweet.decode('utf-8'))
        # print "[elong]", tweet

        # 7. tokenization using tweetNLP
        tweet = ' '.join(twokenize.simpleTokenize(tweet))
        # print "[token]", tweet

        # 8. fix \n char
        tweet = tweet.replace('\n', ' ')
        prccd_item_list.append(tweet.strip())
        # print (tweet)
    return prccd_item_list

'''Main Function'''

def main(csvfile):
    columns = []  # each value in each column is appended to a list

    with open(csvfile, 'rb') as f:
        for line in f:
            columns.append(line)
            #print (line)
    prccd_item_list = process(columns)
    name = os.path.splitext(csvfile)[0]

    with open("%s_prccd.txt" % name, 'wb') as pre_file:
        for s in prccd_item_list:
            pre_file.write(s+'\n')
            print (s)



# ===========
# ==> Run <==
# ===========
main(sys.argv[1])
