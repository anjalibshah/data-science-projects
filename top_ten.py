import sys
import json
import re
import operator

"""
Find the ten most frequently occurring hashtags from twitter data.
"""

twitterData = sys.argv[1] #output.txt

def lines(fp):
    print str(len(fp.readlines()))
    
def get_sentiments():
    """
	Creates in-memory JSON objects from a tweet file ``fp``.
    """
    tweets = []
    for line in open(twitterData):
        tweets.append(json.loads(line))
    return tweets

def get_top_ten_hashtags():
    """
	Loop through list of tweets to obtain hashtag text
	Add hashtag text to a dictionary and update count
	each time the same hashtag text is ecountered across
	all tweets
    """
    
    tweets = get_sentiments()
    
    pattern = re.compile(r'\w+')
    hashtag_freq = {}
        
    for t in tweets:
        if 'entities' in t and 'hashtags' in t['entities'] and len(t['entities']['hashtags']) != 0 and 'delete' not in t.keys():
            for item in t['entities']['hashtags']:
                hash_text = item['text'].encode('utf8')
                hash_text = hash_text.rstrip('?:!.,;"!@')
                hash_text = hash_text.replace("\n", "")
            
                if hash_text in hashtag_freq.keys():
                    tot_freq = hashtag_freq.get(hash_text) + 1
                    hashtag_freq[hash_text] = tot_freq
                else:
                    hashtag_freq.update({hash_text:1})
        else:
            continue
    
    # Sort the hash tags to have those with highest frequency count on top
    top_ten_hashtags = sorted(hashtag_freq.iteritems(), key=operator.itemgetter(1), reverse = True)[0:10]
    
    for item in top_ten_hashtags:
        print item[0], item[1]
                   
def main():
    get_top_ten_hashtags()
    
if __name__ == '__main__':
    main()
