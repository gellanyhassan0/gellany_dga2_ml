from sklearn import *
import pandas as pd
import numpy as np
import pylab
import tldextract
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

pylab.rcParams['figure.figsize'] = (14.0, 5.0)
pylab.rcParams['axes.grid'] = True





class dga():
    
    def __init__(self, uri =None):
               
               self.uri = uri
    
              

    def domain_extract(self):
              ext= tldextract.extract(self.uri)
              if (not ext.suffix):
                    return np.nan
              else:
                    return ext.domain


    def print_domain_extract(self):
            
             
            alexa_dataframe['domain'] = [dga(uri = uri).domain_extract() for uri in alexa_dataframe['uri']]
            del alexa_dataframe['rank']
            del alexa_dataframe['uri']
            print(alexa_dataframe.head())

    def preprocessing(self):
            

            # It's possible we have NaNs from blanklines or whatever
            global alexa_dataframe2
            alexa_dataframe2 = alexa_dataframe.fillna(0.0, inplace=True)
            alexa_dataframe2 = alexa_dataframe.drop_duplicates()

            # Set the class
            alexa_dataframe2['class'] = 'legit'

            # Shuffle the data (important for training/testing)
            alexa_dataframe2 = alexa_dataframe2.reindex(np.random.permutation(alexa_dataframe2.index))
            alexa_total = len(alexa_dataframe2)
            print('Total Alexa domains %d' % alexa_total)

            # Hold out 10%
            hold_out_alexa = alexa_dataframe2[:int(alexa_total*10/100)]
            alexa_dataframe2 = alexa_dataframe2[:int(alexa_total*90/100)]

            print('Number of Alexa domains: %d' % alexa_dataframe2.shape[0])


            
alexa_dataframe = pd.read_csv('data/alexa_100k.csv', names=['rank', 'uri'], header=None, encoding='utf-8')
print(alexa_dataframe.head())
print(alexa_dataframe.tail())            
dga().print_domain_extract()
dga().preprocessing()

