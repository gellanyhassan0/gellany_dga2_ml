import pandas as pd
import numpy as np
import pylab
import tldextract
import math
from collections import Counter
from sklearn import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
import operator
import warnings
import pickle
import argparse
warnings.filterwarnings("ignore", category=DeprecationWarning)

pylab.rcParams['figure.figsize'] = (14.0, 5.0)
pylab.rcParams['axes.grid'] = True



alexa_final = pd.DataFrame()
dga_final = pd.DataFrame()

class dga():
    
    def __init__(self, uri =None, dataframe = None, type = '', dga = False , s = None, domain = '', data =None , encode = None, domain_without_sub = '', domain_entropy = None, domain_length =None):
               
               self.uri = uri
               self.dataframe = dataframe
               self.df = type
               self.dga = dga
               self.s = s
               self.domain = domain
               self.data = data
               self.encode = encode
               self.domain_without_sub = domain_without_sub
               self.domain_entropy = domain_entropy
               self.domain_length = domain_length
               
               

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
            
            if self.dga == False:
                  self.df['class'] = 'legit'
            if self.dga == True:
                  self.df['class'] = 'dga'

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
            

    def entropy(self):
            p, lns = Counter(self.s), float(len(self.s))
            return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

    def plot_cm(self):
    
            # Compute percentanges
            percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T)  # Derp, I'm sure there's a better way
            
            print('Confusion Matrix Stats')
            for i, label_i in enumerate(labels):
                  for j, label_j in enumerate(labels):
                        print("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))

            # Show confusion matrix
            # Thanks kermit666 from stackoverflow :)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.grid(b=False)
            cax = ax.matshow(percent, cmap='coolwarm')
            pylab.title('Confusion matrix of the classifier')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            pylab.xlabel('Predicted')
            pylab.ylabel('True')
            pylab.show()

    def core(self):
            global all_domains
            all_domains = pd.concat([alexa_final, dga_final], ignore_index=True)
            all_domains['length'] = [len(str(x)) for x in all_domains['domain']]
            all_domains = all_domains[all_domains['length'] > 6]
            all_domains['entropy'] = [dga(s = x).entropy() for x in all_domains['domain']]
            print(all_domains.head())
            print(all_domains.tail())
            high_entropy_domains = all_domains[all_domains['entropy'] > 4]
            print('Num Domains above 4 entropy: %.2f%% %d (out of %d)' % (100.0*high_entropy_domains.shape[0]/all_domains.shape[0],high_entropy_domains.shape[0],all_domains.shape[0]))
            print("Num high entropy legit: %d" % high_entropy_domains[high_entropy_domains['class']=='legit'].shape[0])
            print("Num high entropy DGA: %d" % high_entropy_domains[high_entropy_domains['class']=='dga'].shape[0])
            print(high_entropy_domains[high_entropy_domains['class']=='legit'].head())
            print(high_entropy_domains[high_entropy_domains['class']=='dga'].head())
            X = all_domains[['length', 'entropy']]
            y = np.array(all_domains['class'].tolist())
            clf = RandomForestClassifier(n_estimators=20)
            scores = cross_val_score(clf, X, y, cv=5, n_jobs=4)
            print(scores)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            global labels
            labels = ['legit', 'dga']
            global cm
            cm = confusion_matrix(y_test, y_pred)
            dga().plot_cm()
            
    def ordinal_encode(self):
                     
                     self.data[self.encode][self.data[self.encode].isna()] = 'NaN'
                     
                     ord_enc = OrdinalEncoder()
                     ord_enc = ord_enc.fit(self.data[self.encode])
                     self.data[self.encode] = ord_enc.transform(self.data[self.encode])



    def evaluate_models(self):
                     # Select example rows to display.
                     predictions.select("Domain","Type","features","predictedLabel").show(50)

                     # Select (prediction, true label) and compute test error
                     evaluator = MulticlassClassificationEvaluator(
                     labelCol="indexed_type", predictionCol="prediction", metricName="accuracy")
                     accuracy = evaluator.evaluate(predictions)
                     print("Test Error = %g" % (1.0 - accuracy))  


    def entropy_check(self):
                  
                    
                   
                    global l
                    l = self.domain_without_sub
                    
                    if not dga().avg_transition_prob() > threshold:
                           print("Domain %s is DGA!" % args.domain)
                           return 1
                    else: 
                           print("Domain %s is clean!" % args.domain)
                           return 0

    def avg_transition_prob(self):
                    #s = l
                    #global string
                    try :
                         s = l
                    except :
                         s = self.s
                    
                    
                    """ Return the average transition prob from l through log_prob_mat. """
                    log_prob = 0.0
                    transition_ct = 0
                    global n
                    n = 2
                    global accepted_chars 
                    accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
                    global pos
                    pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)]) 
                    #print("pos", pos)
                    
                    i = 0
                    for i in range(0, len(s), 1):
                       
                        try :
                           a, b = s[i:i+2]
                           #print("s[i:i+2]", s[i:i+2])
                           #print("a, b", a, b)
                           #print("[pos[a]][pos[b]]",[pos[a]]+[pos[b]])
                           log_prob += model_mat[pos[a]][pos[b]]
                           transition_ct += 1
                           print("transition_ct", transition_ct)
                           print("log_prob", log_prob)
                          
                        except Exception:
                                        pass


                    #print("transition_ct_final", transition_ct)
                    #print("log_prob_final", log_prob)
                    # The exponentiation translates from log probs to probs.
                    #print("math.exp(log_prob / (transition_ct or 1)", math.exp(log_prob / (transition_ct or 1)))
                    return math.exp(log_prob / (transition_ct or 1))


  
    def test_it(self):
                     all_domains['class'] = all_domains['class'].map({'legit':0, 'dga':1})
                     all_domains['domain'] = all_domains['domain'].astype(str)
       
                     #all_domains['domain_new'] = all_domains['domain'].apply(lambda x : 0.0 if dga(domain_without_sub = x).gellany_avg_transition_prob() > threshold else 1.0)
                     all_domains['domain_new'] = all_domains['domain'].apply(lambda x : 0.0 if dga(s = x).avg_transition_prob() > threshold else 1.0)

                     print(all_domains.dtypes)
                     print(all_domains.head())
                     all_domains.to_csv("all_domains2.csv")
                    
                     
                     X_train, X_test, y_train, y_test = train_test_split(all_domains[["domain_new","length","entropy"]], all_domains["class"] , test_size=0.2, random_state=42)
                     
                     model = RandomForestClassifier()
                     model.fit(X_train,y_train)
                     model_pred = model.predict(X_test)

                     print('Accuracy score for', model, format(accuracy_score(y_test, model_pred)))
               
                     print('F1 score for' ,model, format(f1_score(y_test, model_pred)))
    
def main():
                    
                     if args.domain :
                                 dga(domain_without_sub = args.domain).entropy_check()
                           
                     else:


                                
                                 global alexa_dataframe
                                 global dga_dataframe
 
                                 alexa_dataframe = pd.read_csv(args.file_normal, names=['rank', 'uri'], header=None, encoding='utf-8')
                                 dga_dataframe = pd.read_csv(args.file_dga, names=['raw_domain'], header=None, encoding='utf-8')
                                 dga_dataframe['domain'] = dga_dataframe.applymap(lambda x: x.split('.')[0].strip().lower())
                                 word_dataframe = pd.read_table('data/words.txt', names=['word'])
                                 #word_dataframe = word_dataframe.fillna(0.0, inplace=True)
                                 print(word_dataframe)

                                 #alexa_dataframe2 = pd.DataFrame()
                                 print(alexa_dataframe.head())
                                 print(alexa_dataframe.tail())            
                                 dga(dataframe = alexa_dataframe).print_domain_extract()
                                 dga(dataframe = alexa_dataframe, type = 'alex', dga = False).preprocessing()
                                 dga(dataframe = dga_dataframe, type = 'dga', dga= True).preprocessing()
                                 print(dga_final.head())
                                 dga().core()

                                 dga().test_it()
      
 

            

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-d", "--domain", help="Domain to check")
my_parser.add_argument("-fn", "--file_normal", help="File with normal. One per line")
my_parser.add_argument("-fd", "--file_dga", help="File with dga. One per line")
args = my_parser.parse_args()

model_data = pickle.load(open('gib_model.pki', 'rb'))
model_mat = model_data['mat']
threshold = float(model_data['thresh'])
print('threshold', threshold)


main()
   
            
      
            
            



        
