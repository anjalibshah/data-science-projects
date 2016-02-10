#!/usr/bin/python -tt

import numpy as np
import scipy.stats as st
import sys
import initializedataframe as df
import matplotlib.pyplot as plt


def initxandyvalues(customer_id_string):
  
  dfvolumescore = df.initdf(customer_id_string)
  
  x = dfvolumescore['patientvolume'].values
  y = dfvolumescore['nps'].values
  y[y<90]=0
  y[y>=90]=1
 
  return x, y


def performtestforsignificance(x_val, y_val):

  x, y = x_val, y_val
 
  two_sample = st.mannwhitneyu(x[y==1], x[y==0])

  print "The U-statistic is %.3f and the p-value is %.3f." % two_sample

  return two_sample.pvalue


def plotscoredistributions(x_val, y_val):
  
  x, y = x_val, y_val

  plt.hist(x[y==0], color='red', ls='dashed', bins=5, range=(0,50), fc=(1,0,0,0.5), lw=2.5, edgecolor='red', label = 'Bad Scores')
  plt.hist(x[y==1], color='blue', ls='dashed', bins=5, range=(0,50), fc=(0,0,1,0.5), lw=2.5, edgecolor='blue', label = 'Good Scores')

  plt.xlabel('Patients Seen by Provider Per Day', fontsize=15)
  plt.ylabel('Frequency of Net Promoter Score', fontsize=15)

  legend = plt.legend(loc='upper right', prop={'size':15})

  plt.show()

def getintervalforgoodscores(x_val, y_val):

  x, y = x_val, y_val

  print "Interval for Good Scores: ",
  
  lower_endpt, upper_endpt = st.t.interval(0.95, len(x[y==1])-1, loc=np.mean(x[y==1]), scale=st.sem(x[y==1]))

  print int(np.trunc(lower_endpt)), " - ", int(np.trunc(upper_endpt)), " patients per provider in a day"


def getintervalforbadscores(x_val, y_val):

  x, y = x_val, y_val

  print "Interval for Bad Scores: ",
  
  lower_endpt, upper_endpt = st.t.interval(0.95, len(x[y==0])-1, loc=np.mean(x[y==0]), scale=st.sem(x[y==0]))
 
  print int(np.trunc(lower_endpt)), " - ", int(np.trunc(upper_endpt)), " patients per provider in a day"


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  
  # Main function that calls the relevant methods to find interval cut-offs for good and bad scores. Take customer_id string as argument
  x_val, y_val = initxandyvalues("1")

  print "\n"

  pvalue = performtestforsignificance(x_val, y_val)

  if pvalue > 0.05:
    print "Results of the Mann-Whitney U test are not significant. We cannot proceed with calculating score intervals."
    sys.exit()
  
  getintervalforgoodscores(x_val, y_val)

  getintervalforbadscores(x_val, y_val)

  print "\n"  

  plotscoredistributions(x_val, y_val)

