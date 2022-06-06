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
    
    def __init__(self, uri =None, dataframe = None):
               
               self.uri = uri
               self.dataframe = dataframe
              

    def domain_extract(self):
              ext= tldextract.extract(self.uri)
              if (not ext.suffix):
                    return np.nan
              else:
                    return ext.domain


    def print_domain_extract(self):
            
             
            self.dataframe['domain'] = [dga(uri = uri).domain_extract() for uri in self.dataframe['uri']]
            del alexa_dataframe['rank']
            del alexa_dataframe['uri']
            print(self.dataframe.head())

    def preprocessing(self):
            # It's possible we have NaNs from blanklines or whatever
            #global df
            df = self.dataframe.fillna(0.0, inplace=True)
            df = self.dataframe.drop_duplicates()

            # Set the class
            df['class'] = 'legit'

            # Shuffle the data (important for training/testing)
            df = df.reindex(np.random.permutation(df.index))
            alexa_total = len(df)
            print('Total Alexa domains %d' % alexa_total)

            # Hold out 10%
            hold_out_alexa = df[:int(alexa_total*10/100)]
            alexa_dataframe2 = df[:int(alexa_total*90/100)]

            print('Number of Alexa domains: %d' % df.shape[0])
            print(df.head())

        
alexa_dataframe = pd.read_csv('data/alexa_100k.csv', names=['rank', 'uri'], header=None, encoding='utf-8')
dga_dataframe = pd.read_csv('data/dga_domains.txt', names=['raw_domain'], header=None, encoding='utf-8')
dga_dataframe['domain'] = dga_dataframe.applymap(lambda x: x.split('.')[0].strip().lower())

print(alexa_dataframe.head())
print(alexa_dataframe.tail())            
dga(dataframe = alexa_dataframe).print_domain_extract()
dga(dataframe = alexa_dataframe).preprocessing()
dga(dataframe = dga_dataframe).preprocessing()
