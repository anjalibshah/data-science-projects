import pandas as pd
import pymysql as mdb
import numpy as np
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm
import datetime
from dateutil import relativedelta
from dbconnection import readdbconfig
import warnings


def getstartdate(df):
 
  dates = df.index.values
  startdate = pd.Timestamp.date(pd.to_datetime(dates[0:1], format='%Y-%m-%d').to_pydatetime()[0])
  startdate = '{0.year}-{0.month}-{0.day}'.format(startdate)
  return startdate
  

def getenddate(df):

  dates = df.index.values
  enddate = pd.Timestamp.date(pd.to_datetime(dates[-1:], format='%Y-%m-%d').to_pydatetime()[0])
  enddate = '{0.year}-{0.month}-{0.day}'.format(enddate)
  return enddate


def initdfmonth(customer_id_string, site_id_string):
  # Connect to DB and initialize dataframe with required information

  host, user, passwd, db = readdbconfig()
  
  try:
  
    calibcon = mdb.connect(host, user, passwd, db)

    queryString = 'select * from patient_visit where customer_id in ("' + customer_id_string + '") and site_id in ("' + site_id_string + '")'

    dfsite = pd.read_sql(queryString, con=calibcon)

  except:
    print "An unexpected error occurred", sys.exc_info()[0]
    raise

  else:

    dfvisits = dfsite.loc[:,['visit_id','date_seen']]
    dfvisits['date_seen_index'] = pd.to_datetime(dfvisits['date_seen'], format='%Y-%m-%d')
    indexed_dfvisits = dfvisits.set_index(dfvisits['date_seen_index'])
    indexed_dfvisits = indexed_dfvisits[indexed_dfvisits.visit_id > 0]
    dfvolume = indexed_dfvisits.groupby(indexed_dfvisits.index).count()
    dfvolume = dfvolume.rename(columns = {'visit_id':'patient_volume_by_day'})
    
    firstmonth = getstartdate(dfvolume)
    lastmonth = getenddate(dfvolume)
    
    idx = pd.date_range(firstmonth, lastmonth)
    dfvolume = dfvolume.reindex(idx, fill_value=0)
    
    dfvolumemonth = dfvolume.resample('M', how='sum')
    
    # Removing last month's information due to insufficient data
    dfvolumemonth = dfvolumemonth.iloc[0:-1]
    return dfvolumemonth
    
def mean_absolute_percentage_err(y, yhat):
  
  return np.mean((np.abs(y.sub(yhat)) / y)) * 100

# Model fitness is based on values passed to order param.
# 2,0,0 work well for this customer and site. For others, try different numbers and pick the ones that give least values for error, AIC and BIC
def fitarimamodel(dfvolumemonth):
  
  model = sm.tsa.ARIMA(dfvolumemonth['patient_volume_by_day'].iloc[0:], order=(2, 0, 0))  
  results = model.fit(disp=-1)  
  dfvolumemonth['Forecast'] = results.fittedvalues  
  print "Mean Absolute Percentage Error: %.3f " % mean_absolute_percentage_err(dfvolumemonth['patient_volume_by_day'], dfvolumemonth['Forecast']),"%"
  print "AIC: ", results.aic
  print "BIC: ", results.bic
  print "\n"
  return results, dfvolumemonth

def getnextmonth(dfvolumemonth):

  lastmonth = dfvolumemonth.index.values
  lastmonth = pd.to_datetime(lastmonth[-1:], format='%Y-%m-%d')
  nextmonth = pd.Timestamp.date(lastmonth.to_pydatetime()[0]) + relativedelta.relativedelta(months=1)
  return nextmonth

def forecastvolumefornextmonth(results, dfvolumemonth):
  
  nextmonth = getnextmonth(dfvolumemonth)
  nextmonth = '{0.year}-{0.month}-{0.day}'.format(nextmonth)
  predict_volume_next_month = results.predict(nextmonth, nextmonth, dynamic=False)
  print "Expected volume next month:"
  print predict_volume_next_month
  print "\n"
  
def plotforecastedvolume(results, dfvolumemonth):

  train = dfvolumemonth[0:len(dfvolumemonth)-2]
  test = dfvolumemonth[len(dfvolumemonth)-3:len(dfvolumemonth)]
  
  trainstartdate = getstartdate(train)
  trainenddate = getenddate(train)
  
  teststartdate = getstartdate(test)
  testenddate = getenddate(test)
  
  predict_volume_in_sample = results.predict(start=trainstartdate, end=trainenddate, dynamic=False)
  predict_volume_out_sample = results.predict(start=teststartdate, end=testenddate, dynamic=False)

  plt.figure(figsize = (16,12))
  plt.plot(dfvolumemonth.index, dfvolumemonth['patient_volume_by_day'], label='Actual')
  plt.plot(train.index, predict_volume_in_sample.iloc[0:].values, label='In-sample Forecast')
  plt.plot(test.index, predict_volume_out_sample.iloc[0:].values, label='Out-sample Forecast')

  plt.legend(loc='upper left', prop={'size':14})

  labels = ['Jul 2014', 'Aug 2014','Sep 2014','Oct 2014','Nov 2014','Dec 2014','Jan 2015', 'Feb 2015', 'Mar 2015', 'Apr 2015', 'May 2015', 'Jun 2015', 'Jul 2015', 'Aug 2015', 'Sep 2015', 'Oct 2015', 'Nov 2015', 'Dec 2015']

  plt.xticks(dfvolumemonth.index, labels, rotation = 'vertical', fontsize=12)
  plt.yticks(fontsize=12)

  plt.ylabel('Patient Volume By Month', fontsize=22)
  
  plt.show()

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  
  warnings.filterwarnings("ignore")
  
  dfvolumemonth = initdfmonth("1","2")
  
  results, dfvolumemonth = fitarimamodel(dfvolumemonth)
  forecastvolumefornextmonth(results, dfvolumemonth)
  plotforecastedvolume(results, dfvolumemonth)

    
