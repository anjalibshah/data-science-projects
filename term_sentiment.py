import sys
import json
import re

"""
Calculates the sentiment score for terms in Tweets that are not already
in the AFINN.txt sentiment score file. The equation to calculate a 
sentiment score was derived from 
http://www.cs.cmu.edu/~rbalasub/publications/oconnor_balasubramanyan_routledge_smith.icwsm2010.tweets_to_polls.pdf.
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

def get_non_sentiment_score():
    
    non_sentiment_score = {}
    tweets = get_sentiments()
    score_dict = build_score_dict()
    parsed_tweets = get_parsed_tweets(tweets)
        
    for index in range(len(parsed_tweets)):
        
        tweet_linetext = parsed_tweets[index]
        pos_cnt = 0
        neg_cnt = 0
        term_score = 0
        non_sentiments = []
        
        for word in tweet_linetext:
            
            word = word.rstrip('?:!.,;"!@')
            word = word.replace("\n", "")
            
            # Obtain counts of positive and negative sentiment words            
            if word.encode('utf8') in score_dict.keys():
                value = score_dict.get(word.encode('utf8'))
                if value < 0: neg_cnt = neg_cnt + 1
	        elif value > 0: pos_cnt = pos_cnt + 1
            else:
                non_sentiments.append(word) # Collect non-sentiments in a list
        
        # Obtain ratio of positive to negative sentiments for each tweet            
	if neg_cnt != 0:
	    term_score = float(pos_cnt)/neg_cnt 
	else: 
	    term_score = pos_cnt 
	
	# Loop through the list of non-sentiments and assign sentiment score based on tweet sentiment ratio                
        for word in non_sentiments:
            if word.encode('utf8') in non_sentiment_score.keys():
                tot_term_score = term_score + non_sentiment_score.get(word.encode('utf8'))
                non_sentiment_score[word.encode('utf8')] = tot_term_score
            else:
                non_sentiment_score.update({word.encode('utf8'):term_score})
                                
    for key, value in non_sentiment_score.items():
        print "%s %.2f" % (key, value)
            

def main():
    get_non_sentiment_score()

if __name__ == '__main__':
    main()
