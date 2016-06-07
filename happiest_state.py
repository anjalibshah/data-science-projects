import sys
import json
import re
import operator

"""
Find happiest US state based on tweet sentiments.
"""

sentimentData = sys.argv[1] #AFIN-111.txt
twitterData = sys.argv[2] #output.txt

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

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

def get_happiest_state():
    """
	Parses a list of tweets to obtain location information specific to US state
	from each tweet.
	For each such tweet, parse it, split the ``text`` of the tweet
	into tokens based on a regular expression, calculate the sentiment score.
	Return the state with the highest sentiment score
    """
    
    tweets = get_sentiments()
    score_dict = build_score_dict()
    pattern = re.compile(r'\w+')
    state_senti_score = {}
        
    for t in tweets:
        if 'user' in t and 'location' in t['user'] and t['user']['location'] != None:
            word_loc = t['user']['location'].encode('utf8')
            
            # Obtain 2 letter abbreviated state name
            if word_loc.find(", ") != -1:
                words = word_loc.split(", ")
                for word in words:
                    if word in states.keys(): 
                        state = word
                        break
                    elif word in states.values():
                        state = [k for k, v in states.iteritems() if v == word][0]
                        break
            else:
                if word_loc == 'None': continue
                if word_loc in states.keys():
                    state = word_loc
                elif word_loc in states.values():
                    state = [k for k, v in states.iteritems() if v == word_loc][0]
                else:
                    continue
            
            if 'text' not in t.keys():
                continue

	    # Obtain a list of words
	    else:
	        tweet_linetext = pattern.findall(t['text'])
	        sum_score = 0
	        
	        # Obtain sentiment score for each tweet given the list of words
	        for word in tweet_linetext:
            
                    word = word.rstrip('?:!.,;"!@')
                    word = word.replace("\n", "")
              
                    if word.encode('utf8') in score_dict.keys():
                        sum_score += score_dict.get(word.encode('utf8'))
                    else:
                        sum_score += 0
            if state in state_senti_score.keys():
                tot_state_score = float(sum_score + state_senti_score.get(state))/2
                state_senti_score[state] = tot_state_score
            else:
                state_senti_score.update({state:sum_score})
        else:
            continue
    # Get the state with the maximum sentiment score
    print max(state_senti_score.iteritems(), key=operator.itemgetter(1))[0]
                   
def main():
    get_happiest_state()

if __name__ == '__main__':
    main()
