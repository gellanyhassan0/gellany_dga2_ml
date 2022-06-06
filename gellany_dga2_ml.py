from sklearn import *
import pandas as pd
import numpy as np
import pylab
import tldextract
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

pylab.rcParams['figure.figsize'] = (14.0, 5.0)
pylab.rcParams['axes.grid'] = True



alexa_final = pd.DataFrame()
dga_final = pd.DataFrame()

class dga():
    
    def __init__(self, uri =None, dataframe = None, type = '', dga = False):
               
               self.uri = uri
               self.dataframe = dataframe
               self.df = type
               self.dga = dga
               
              

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
           

            self.df = self.dataframe.fillna(0.0, inplace=True)
            self.df = self.dataframe.drop_duplicates()

            # Set the class
            self.df['class'] = 'legit'

            # Shuffle the data (important for training/testing)
            self.df = self.df.reindex(np.random.permutation(self.df.index))
            alexa_total = len(self.df)
            print('Total Alexa domains %d' % alexa_total)

            # Hold out 10%
            hold_out_alexa = self.df[:int(alexa_total*10/100)]
            alexa_dataframe2 = self.df[:int(alexa_total*90/100)]

            print('Number of Alexa domains: %d' % self.df.shape[0])
            print(self.df.head())
            
            if self.dga == False:
                  global alexa_final
                  alexa_final = alexa_final.append(self.df)
            if self.dga == True:
                  global dga_final
                  dga_final = dga_final.append(self.df)

            
            
            
             
           
            
      
            
            


        
alexa_dataframe = pd.read_csv('data/alexa_100k.csv', names=['rank', 'uri'], header=None, encoding='utf-8')
dga_dataframe = pd.read_csv('data/dga_domains.txt', names=['raw_domain'], header=None, encoding='utf-8')
dga_dataframe['domain'] = dga_dataframe.applymap(lambda x: x.split('.')[0].strip().lower())
#alexa_dataframe2 = pd.DataFrame()
print(alexa_dataframe.head())
print(alexa_dataframe.tail())            
dga(dataframe = alexa_dataframe).print_domain_extract()
dga(dataframe = alexa_dataframe, type = 'alex').preprocessing()
dga(dataframe = dga_dataframe, type = 'dga').preprocessing()
all_domains = pd.concat([alexa_final, dga_final], ignore_index=True)
print(all_domains.head())