import sys
import json
import re

"""
Parses Tweet objects (https://dev.twitter.com/docs/platform-objects/tweets) 
obtained for Twitter's streaming API, calculate the
frequency for every word that appears in the tweets.
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

def get_parsed_tweets(tweets):
    """
	Parses a list of tweets, splitting the ``text`` of the tweet
	into tokens based on a regular expression.
    """
    pattern = re.compile(r'\w+')
    parsed_tweets = []
    
    for t in tweets:
        if 'text' not in t.keys():
            continue

	# Obtain a list of words
        words = pattern.findall(t['text'])
	parsed_tweets.append(words)

    return parsed_tweets

def get_term_freq():
    """
    Loop through each tweet and for each individual tweet, add up the total number
    of times the word appears in the tweet.
    Parse through all tweets and sum up the total number of words in all tweets
    as well as the total number of times each word appears in all tweets combines
    Use the above two pieces of information to get frequency of each word across all tweets
    
    """
    
    total_freq = {}
    tweets = get_sentiments()
    parsed_tweets = get_parsed_tweets(tweets)
    tot_all_terms = 0
    
    for index in range(len(parsed_tweets)):
        tweet_freq = {}
        tweet_linetext = parsed_tweets[index]
        
        for word in tweet_linetext:
            
            word = word.rstrip('?:!.,;"!@')
            word = word.replace("\n", "")
                      
            if word.encode('utf8') in tweet_freq.keys():
                term_freq = tweet_freq.get(word.encode('utf8')) + 1
                tweet_freq[word.encode('utf8')] = term_freq
            else:
                tweet_freq[word.encode('utf8')] = 1
            
        for key in tweet_freq.keys():
            if key in total_freq.keys():
                tot_freq = tweet_freq.get(key) + total_freq.get(key)
                total_freq[key] = tot_freq
            else:
                total_freq.update({key:1})
    
    tot_all_terms = sum([value for key,value in total_freq.items()])
                    
    for key, value in total_freq.items():
        term_freq = float(value)/tot_all_terms
        print "%s %.10f" % (key, term_freq)

def main():
    get_term_freq()

if __name__ == '__main__':
    main()
