#!/usr/bin/python -tt

import pandas as pd
import pymysql as mdb
from dbconnection import readdbconfig


def initdf(customer_id_string):
  # Connect to DB and initialize dataframe with required information

  host, user, passwd, db = readdbconfig()
  
  try:
  
    calibcon = mdb.connect(host, user, passwd, db)

    queryString = 'select * from patient_visit where customer_id in ("' + customer_id_string + '")'

    dfall = pd.read_sql(queryString, con=calibcon)

  except:
    print "An unexpected error occurred", sys.exc_info()[0]
    raise

  else:
  
    # Get information to count patient visits per day
    dfcntpatients = dfall.loc[:,['visit_id', 'provider_id', 'date_seen']]

    dfcountpatients = dfcntpatients.set_index(['date_seen'])

    dfcountpatients = dfcountpatients[dfcountpatients.visit_id > 0]

    # Remove rows without any visit ID
    dfcountpatients = dfcountpatients[dfcountpatients.provider_id != '']

    # Get raw scores where available
    dfcntscore = dfall.loc[:,['score', 'provider_id', 'date_seen']]

    dfcntscore = dfcntscore.dropna(axis=0)

    dfcntscore = dfcntscore[dfcntscore.score != '']

    dfcountscore = dfcntscore.set_index(['date_seen'])

    dfcountscoretotalcount = dfcountscore.groupby([dfcountscore.index, 'provider_id']).count()

    dfcountscoreatt = dfcountscore[dfcountscore['score'].isin(['9','10'])]

    dfcountscoredet = dfcountscore[dfcountscore['score'].isin(['0','1','2','3','4','5','6'])]

    dfcountscore['score'] = dfcountscore['score'].astype(int)

    dfcountscoremean = dfcountscore.groupby([dfcountscore.index, 'provider_id']).mean()

    dfcountscoreatt1 = dfcountscoreatt.groupby([dfcountscoreatt.index, 'provider_id']).count()

    dfcountscoredet1 = dfcountscoredet.groupby([dfcountscoredet.index, 'provider_id']).count()

    dfcountscoreatt1 = dfcountscoreatt1.rename(columns = {'score':'scoreatt'})
    dfcountscoredet1 = dfcountscoredet1.rename(columns = {'score':'scoredet'})

    dfnpsscore = pd.concat([dfcountscoretotalcount, dfcountscoreatt1, dfcountscoredet1], axis=1, join_axes=[dfcountscoretotalcount.index])

    dfnpsscore['scoreatt'] = dfnpsscore['scoreatt'].fillna(0)
  
    dfnpsscore['scoredet'] = dfnpsscore['scoredet'].fillna(0)

    dfnpsscore['nps'] = (dfnpsscore.scoreatt - dfnpsscore.scoredet)/dfnpsscore.score * 100

    dfcountpatientstotalcount = dfcountpatients.groupby([dfcountpatients.index, 'provider_id']).count()

    dfcountpatientstotalcount = dfcountpatientstotalcount.rename(columns = {'visit_id':'patientvolume'})

    dfvolumescore = pd.concat([dfcountpatientstotalcount, dfnpsscore], axis=1, join_axes=[dfcountpatientstotalcount.index])

    return dfvolumescore
