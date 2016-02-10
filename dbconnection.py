#!/usr/bin/python -tt

from BeautifulSoup import BeautifulSoup
import sys

def readdbconfig():

  try:
    with open("config.xml") as f:
        content = f.read()

  except:
    print "An unexpected error occurred", sys.exc_info()[0]
    raise

  else:
    y = BeautifulSoup(content)
    return y.mysql.host.contents[0], y.mysql.user.contents[0], y.mysql.passwd.contents[0], y.mysql.db.contents[0]

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  print readdbconfig()

