import sys
import json
import re

"""
Derives sentiment of each tweet downloaded from twitter.
"""

sentimentData = sys.argv[1] #AFIN-111.txt
twitterData = sys.argv[2] #output.txt

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

def build_score_dict():
    afinnfile = open(sentimentData)
    scores = {} # initialize an empty dictionary
    for line in afinnfile:
        term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
        scores[term] = int(score)  # Convert the score to an integer.
    #print scores.items() # Print every (term, score) pair in the dictionary
    return scores

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

def get_tweet_score():
    """
    Loop through each tweet and for each individual tweet, add up the sentiment 
    score, based on the scores sentinment dictionary.
    
    """
    
    tweet_score = []
    tweets = get_sentiments()
    score_dict = build_score_dict()
    parsed_tweets = get_parsed_tweets(tweets)
    
    for index in range(len(parsed_tweets)):
        sum_score = 0
        tweet_linetext = parsed_tweets[index]
        
        for word in tweet_linetext:
            
            word = word.rstrip('?:!.,;"!@')
            word = word.replace("\n", "")
                      
            if isinstance(word, unicode):
                if word.encode('utf8') in score_dict.keys():
                    sum_score += score_dict.get(word.encode('utf8'))
                else:
                    sum_score += 0
            elif word in score_dict.keys():
                sum_score += score_dict.get(word)
            else:
                sum_score += 0
        
        tweet_score.append(sum_score)
    
    for y in range(len(tweet_score)):
        print tweet_score[y]
    

def main():
    get_tweet_score()

if __name__ == '__main__':
    main()
