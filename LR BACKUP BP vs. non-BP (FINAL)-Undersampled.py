#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import nltk
import pandas
import xlrd
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams
import matplotlib.pyplot as plt
from os import path
from PIL import Image
import numpy as np
from random import sample
import random
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import itertools
import xlwt
from xlwt import Workbook
import xlsxwriter 
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from collections import Counter

from textblob import TextBlob, Word
import math
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
from sklearn.datasets import make_classification
import imblearn
from imblearn.under_sampling import RandomUnderSampler


# In[2]:


def processed_abstract(abstract):   
# This is a comment: the 1st abstract value is now stored in the variable abstract
#- process it and store the tokens in some variable
    NUM_result1 = re.sub(r'\d+','NUMERIC', abstract)
    PUNC_result1 = re.sub(r'[^A-Za-z0-9-/\s]', ' ', NUM_result1)
    PMID_token1 = nltk.word_tokenize(PUNC_result1)
    
    #PMID_token1 = nltk.wordpunct_tokenize(PUNC_result1)
       
    stop_words = set(stopwords.words('english'))

    filtered_sentence1 = [w for w in PMID_token1 if not w.lower() in stop_words]
    filtered_sentence1 = []
    
    for w in PMID_token1:
        if w not in stop_words:
            if w[-1] in [".",",",":", ";"]:
                w = "".join(w[0:-1])
            filtered_sentence1.append(w)
            
    lemmatizer1 = WordNetLemmatizer()
    lemmatized_abstract1 = [lemmatizer1.lemmatize(word) for word in filtered_sentence1]
    listToStr = ' '.join([str(elem) for elem in lemmatized_abstract1])
    

    return listToStr


# In[3]:


#About Set
full_set2 = pd.read_excel(r"C:\Users\jolsh\Documents\REU\BP_nonBP_PMID (with rest categories).xlsx", 
                          sheet_name = "BP Set")
print(full_set2)


# In[4]:


full_set2['ABSTRACT'].fillna("ABSTRACT EMPTY",inplace=True)


# In[5]:


full_set2['processed_abstracts']=(full_set2['TITLE'] + ' ' + full_set2['ABSTRACT']).apply(lambda x:processed_abstract(x))
print(full_set2)


# In[6]:


#Analysis of About Set
string2 = full_set2['processed_abstracts'].str.cat(sep=', ')

def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)

# Lemmatize
sentence = string2
lemma = lemmatize_with_postag(sentence)

tokenize = nltk.word_tokenize(lemma)

#will show just top x most frequent unigram, bigram, and trigrams
unigram_series2 = (pd.Series(nltk.ngrams(tokenize, 1)))[0:]
bigram_series2 = (pd.Series(nltk.ngrams(tokenize, 2)))[0:]

#stores all the unigram, bigram, and trigrams for later weighting
#unigram_series2 = (nltk.ngrams(filtered_sentence, 1))
#bigram_series2 = (nltk.ngrams(filtered_sentence, 2)))


# In[7]:


joined_unigram2 = []
for x in unigram_series2:
    join = ''.join(x)
    joined_unigram2.append(join)
print(len(joined_unigram2))


# In[8]:


joined_bigram2 = []
for x in bigram_series2:
    join = ' '.join(x)
    joined_bigram2.append(join)
print(len(joined_bigram2))


# In[9]:


DF3 = defaultdict(int)
words = joined_unigram2
unique_words = set(words)

oldcount = 0
print(len(set(words)))
for x in full_set2['processed_abstracts']:
    for word in unique_words:
        if len(word) >= 2 and word in x:
            DF3[word] += 1
        else:
            oldcount = DF3.get(word)
            if oldcount is None:
                DF3[word] = 0
            else:
                DF3[word] = DF3[word]
print(len(DF3))


# In[10]:


DF4 = defaultdict(int)
words = joined_bigram2
unique_words = set(words)

oldcount = 0
print(len(set(words)))
for x in full_set2['processed_abstracts']:
    for word in unique_words:
        if len(word) >= 2 and word in x:
            DF4[word] += 1
        else:
            oldcount = DF4.get(word)
            if oldcount is None:
                DF4[word] = 0
            else:
                DF4[word] = DF4[word]
print(len(DF4))


# In[11]:


dfb3 = pd.DataFrame.from_dict(DF3, orient = 'index', columns = ['count2'])
dfb3 = dfb3.reset_index()
rslt_dfb3 = dfb3.sort_values(by = 'count2', ascending = False)


# In[12]:


dfb4 = pd.DataFrame.from_dict(DF4, orient = 'index', columns = ['count2'])
dfb4 = dfb4.reset_index()
rslt_dfb4 = dfb4.sort_values(by = 'count2', ascending = False)


# In[13]:


#can set to top values using integer in print statement
#ordered by document frequency for Background set unigrams use .iloc
print(rslt_dfb3.head(3500))
##ordered by document frequency for Background set bigraams
print(rslt_dfb4.head(1500))


# In[14]:


#Background Set 
full_set = pd.read_excel(r"C:\Users\jolsh\Documents\REU\BP_nonBP_PMID (with rest categories).xlsx", sheet_name = "BP_NONBP for LR")
print(full_set)


# In[15]:


full_set['ABSTRACT'].fillna("ABSTRACT EMPTY",inplace=True)


# In[16]:


full_set['processed_abstracts']=(full_set['TITLE'] + ' ' + full_set['ABSTRACT']).apply(lambda x:processed_abstract(x))
print(full_set)


# In[17]:


DF1 = defaultdict(int)
words = joined_unigram2
unique_words = set(words)
oldcount = 0
for x in full_set['processed_abstracts']:
    for word in unique_words:
        if len(word) >= 2 and word in x:
            DF1[word] += 1
        else:
            oldcount = DF1.get(word)
            if oldcount is None:
                DF1[word] = 0
            else:
                DF1[word] = DF1[word]
print(len(DF1))


# In[18]:


DF2 = defaultdict(int)
print(len(DF2))
words = joined_bigram2
unique_words = set(words)

oldcount = 0
for x in full_set['processed_abstracts']:
    for word in unique_words:
        if len(word) >= 2 and word in x:
            DF2[word] += 1
        else:
            oldcount = DF2.get(word)
            if oldcount is None:
                DF2[word] = 0
            else:
                DF2[word] = DF2[word]
print(len(DF2))


# In[98]:


dfb1 = pd.DataFrame.from_dict(DF1, orient = 'index', columns = ['count1'])
dfb1 = dfb1.reset_index()
dfb1 ['word'] = dfb1['index']
rslt_dfb1 = dfb1.sort_values(by = 'count1', ascending = False)


# In[99]:


dfb2 = pd.DataFrame.from_dict(DF2, orient = 'index', columns = ['count1'])
dfb2 = dfb2.reset_index()
dfb2 ['word'] = dfb2['index']
rslt_dfb2 = dfb2.sort_values(by = 'count1', ascending = False)


# In[100]:


#can set to top values using integer in print statement
#ordered by document frequency for Background set unigrams use .iloc
#dfb1['words'] = dfb1 ['index']
print(rslt_dfb1.head(6000))
##ordered by document frequency for Background set bigraams
#dfb2['words'] = dfb2 ['index']
print(rslt_dfb2.head(7000))


# In[101]:


result1 = pd.concat([dfb1,dfb3], axis=1)
print(result1)


# In[102]:


result2 = pd.concat([dfb2,dfb4], axis=1)
print(result2)


# In[103]:


rslt1 = result1.sort_values(by = 'count2', ascending = False)
print(rslt1)


# In[104]:


rslt2 = result2.sort_values(by = 'count2', ascending = False)
print(rslt2)


# In[105]:


#count 1 is for background set count 2 is for about set, looking at unigrams
dict1 = rslt1.head(3500)
print(dict1)


# In[106]:


#count 1 is for background set count 2 is for about set, looking at bigrams
dict2 = rslt2.head(1500)
print(dict2)


# In[107]:


Na = 1050
Nb = 9690
for index,row in dict1.iterrows():
        dict1.at[index,'scores'] = (((row['count2'])/(Na)) - ((row['count1'])/(Nb))) * np.log (Nb / (row['count1'])) 
print(dict1)


# In[108]:


rslt_score1 = dict1.sort_values(by = 'scores', ascending = False).reset_index(drop = True)
rslt_score1.to_csv('eGiftScoringUnigramBP.tsv',sep = '\t')
print(rslt_score1)

#for use later 
rslt_score_for_fisher1 = dict1.sort_values(by = 'scores', ascending = False).reset_index(drop = True)
print(rslt_score_for_fisher1)


# In[109]:


Na = 1050
Nb = 9690
for index,row in dict2.iterrows():
        dict2.at[index,'scores'] = (((row['count2'])/(Na)) - ((row['count1'])/(Nb))) * np.log (Nb / (row['count1'])) 
print(dict2)


# In[110]:


rslt_score2 = dict2.sort_values(by = 'scores', ascending = False).reset_index(drop = True)
rslt_score2.to_csv('eGiftScoringBigramBP.tsv',sep = '\t')
print(rslt_score2)

#for use in Fisher
rslt_score_for_fisher2 = dict2.sort_values(by = 'scores', ascending = False).reset_index(drop = True)
print(rslt_score_for_fisher2)


# In[111]:


#FISHER STARTS HERE


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[112]:


rslt_score3 = rslt_score_for_fisher1


# In[113]:


rslt_score3["a"] = rslt_score3["count2"]
rslt_score3["b"] = rslt_score3["count1"] - rslt_score3["count2"]
rslt_score3["c"] = 1050 - rslt_score3["count2"]
rslt_score3["d"] = 8640 - (rslt_score3["count1"] - rslt_score3["count2"])
#rslt_score3.to_csv('b.tsv',sep = '\t')


# In[114]:


rslt_score3["a+b"] = rslt_score3["a"] + rslt_score3["b"]
rslt_score3["c+d"] = rslt_score3["c"] + rslt_score3["d"]
rslt_score3["a+c"] = rslt_score3["a"] + rslt_score3["c"]
rslt_score3["b+d"] = rslt_score3["b"] + rslt_score3["d"]
print(rslt_score3)
#(rslt_score3["b"]).to_csv('negative test.tsv',sep = '\t')


# In[115]:


joined_factorial1 = []
for x in rslt_score3["a"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial1.append(factorial)
print((joined_factorial1))


# In[116]:


joined_factorial2 = []
for x in rslt_score3["b"]:
    #joinfact = ''.join(factorial)
        factorial = math.factorial(x)
        joined_factorial2.append(factorial)
print(len(joined_factorial2))


# In[117]:


joined_factorial3 = []
for x in rslt_score3["c"]:
    if x < 0:
        joined_factorial3.append(1)
    else:
        factorial = math.factorial(x)
        #joinfact = ''.join(factorial)
        joined_factorial3.append(factorial)
print(len(joined_factorial3))


# In[118]:


joined_factorial4 = []
for x in rslt_score3["d"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial4.append(factorial)
print(len(joined_factorial4))


# In[119]:


joined_factorial5 = []
for x in rslt_score3["a+b"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial5.append(factorial)
print(len(joined_factorial5))


# In[120]:


joined_factorial6 = []
for x in rslt_score3["c+d"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial6.append(factorial)
print(len(joined_factorial6))


# In[121]:


joined_factorial7 = []
for x in rslt_score3["a+c"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial7.append(factorial)
print(len(joined_factorial7))


# In[122]:


joined_factorial8 = []
for x in rslt_score3["b+d"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial8.append(factorial)
print(len(joined_factorial8))


# In[123]:


n = math.factorial(9690)


# In[124]:


fisher_list_numerator = []
for x in range(3500):
    numerator = joined_factorial5[x] * joined_factorial6[x] * joined_factorial7[x] * joined_factorial8[x]
    fisher_list_numerator.append(numerator)
print(len(fisher_list_numerator))


# In[125]:


fisher_list_denominator = []

for x in range(3500):
    denominator = joined_factorial1[x] * joined_factorial2[x] * joined_factorial3[x] * joined_factorial4[x] * n 
    fisher_list_denominator.append(denominator)
print(len(fisher_list_denominator))


# In[126]:


fisher_list_score = []
for x in range(3500):
    value = (fisher_list_numerator[x]) / (fisher_list_denominator[x])
    fisher_list_score.append(value)
#print((fisher_list_score))

rslt_score3["Fisher Scores"] = fisher_list_score
#print(rslt_score1)
new_order = rslt_score3.sort_values(by = 'Fisher Scores', ascending = False).reset_index(drop = True)
print(new_order)


# In[127]:


final_scoring = new_order.drop(['a','b', 'c', 'd', 'a+b', 'c+d', 'a+c', 'b+d'],axis = 1)
print(final_scoring)
final_scoring.to_csv('FisherUnigramBP.tsv',sep = '\t')


# In[ ]:





# In[128]:


#Fisher Test Bigram


# In[129]:


rslt_score4 = rslt_score_for_fisher2


# In[130]:


rslt_score4["a"] = rslt_score4["count2"]
rslt_score4["b"] = rslt_score4["count1"] - rslt_score4["count2"]
rslt_score4["c"] = 1050 - rslt_score4["count2"]
rslt_score4["d"] = 8640 - rslt_score4["b"]
#rslt_score4.to_csv('b.tsv',sep = '\t')
#(rslt_score4["b"]).to_csv('negative test.tsv',sep = '\t')


# In[131]:


rslt_score4["a+b"] = rslt_score4["a"] + rslt_score4["b"]
rslt_score4["c+d"] = rslt_score4["c"] + rslt_score4["d"]
rslt_score4["a+c"] = rslt_score4["a"] + rslt_score4["c"]
rslt_score4["b+d"] = rslt_score4["b"] + rslt_score4["d"]
print(rslt_score4)


# In[132]:


joined_factorial9 = []
for x in rslt_score4["a"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial9.append(factorial)
print(len(joined_factorial9))


# In[133]:


joined_factorial10 = []
for x in rslt_score4["b"]:
    #joinfact = ''.join(factorial)
        factorial = math.factorial(x)
        joined_factorial10.append(factorial)
print(len(joined_factorial10))


# In[134]:


joined_factorial11 = []
for x in rslt_score4["c"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial11.append(factorial)
print(len(joined_factorial11))


# In[135]:


joined_factorial12 = []
for x in rslt_score4["d"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial12.append(factorial)
print(len(joined_factorial12))


# In[136]:


joined_factorial13 = []
for x in rslt_score4["a+b"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial13.append(factorial)
print(len(joined_factorial13))


# In[137]:


joined_factorial14 = []
for x in rslt_score4["c+d"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial14.append(factorial)
print(len(joined_factorial14))


# In[138]:


joined_factorial15 = []
for x in rslt_score4["a+c"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial15.append(factorial)
print(len(joined_factorial15))


# In[139]:


joined_factorial16 = []
for x in rslt_score4["b+d"]:
    factorial = math.factorial(x)
    #joinfact = ''.join(factorial)
    joined_factorial16.append(factorial)
print(len(joined_factorial16))


# In[140]:


fisher_list_numerator2 = []
for x in range(1500):
    numerator2 = joined_factorial13[x] * joined_factorial14[x] * joined_factorial15[x] * joined_factorial16[x]
    fisher_list_numerator2.append(numerator2)
print(len(fisher_list_numerator2))


# In[141]:


fisher_list_denominator2 = []

for x in range(1500):
    denominator2 = joined_factorial9[x] * joined_factorial10[x] * joined_factorial11[x] * joined_factorial12[x] * n 
    fisher_list_denominator2.append(denominator2)
print(len(fisher_list_denominator2))


# In[142]:


fisher_list_score2 = []
for x in range(1500):
    value2 = (fisher_list_numerator2[x]) / (fisher_list_denominator2[x])
    fisher_list_score2.append(value2)
#print((fisher_list_score))

rslt_score4["Fisher Scores"] = fisher_list_score2
#print(rslt_score1)
new_order2 = rslt_score4.sort_values(by = 'Fisher Scores', ascending = False).reset_index(drop = True)
print(new_order2)


# In[143]:


final_scoring2 = new_order2.drop(['a','b', 'c', 'd', 'a+b', 'c+d', 'a+c', 'b+d'],axis = 1)
print(final_scoring2)
final_scoring2.to_csv('FisherBigramBP.tsv',sep = '\t')


# In[144]:


print(final_scoring)
print(final_scoring2)


# In[145]:


feature_list = final_scoring.append(final_scoring2, ignore_index = True)
feature_list.sort_values(by = 'Fisher Scores', ascending = True, inplace = True)
feature_list.to_csv('combinedFeatureListBPFisher.tsv',sep = '\t')
print(feature_list)


# In[146]:


feature_list2 = rslt_score1.append(rslt_score2, ignore_index = True)
feature_list2.sort_values(by = 'scores', ascending = False, inplace = True)
feature_list2.to_csv('combinedFeatureListBPeGift.tsv',sep = '\t')
print(feature_list2)


# In[ ]:





# In[162]:


myvocab ={}
count = 0
top = feature_list['word'].head(2700)
top.to_csv('vocabBP.tsv',sep = '\t')
for x in top:
    myvocab[str(x)] = count
    count+=1
#print(myvocab)


# In[163]:


#correct distribution with undersampling
undersample = full_set
undersample.sort_values(by = 'Label', ascending = False, inplace = True)
#print(undersample)
y = undersample[:1000]
#print(y)
x = undersample[1000:]
#print(len(x))
#print(x)
z = x.sample(frac=0.1, replace=True, random_state=1)
#print(z)

frames = [y, z]
new_set = pd.concat(frames)
#print(new_set)
new_set.to_csv('undersample.tsv',sep = '\t')
new_set.sort_values(by = 'docID_Bibliome', ascending = False, inplace = True)
#print(new_set)


# In[164]:


training_data, testing_data = train_test_split(new_set,random_state = 2000, test_size = 0.2)
print(len(training_data))
print(len(testing_data))


# In[165]:


Y_train = training_data['Label'].values
Y_test = testing_data['Label'].values


# In[166]:


def extract_features(df,field,training_data,testing_data,type="binary"):
    if "binary" in type:
        
        # BINARY FEATURE REPRESENTATION
        cv= CountVectorizer(binary=True, max_df=0.9)#my list of top terms
        cv.fit_transform(training_data[field].values)#transform my datafframe field using the list of top terms
        
        train_feature_set=cv.transform(training_data[field].values)
        test_feature_set=cv.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set,cv
  
    elif "counts" in type:
        
        # COUNT BASED FEATURE REPRESENTATION
        cv= CountVectorizer(binary=False, max_df=0.95)
        cv.fit_transform(training_data[field].values)
        
        train_feature_set=cv.transform(training_data[field].values)
        test_feature_set=cv.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set,cv
    
    
    elif "tf-idf" in type:    
        
        # TF-IDF BASED FEATURE REPRESENTATION
        tfidf_vectorizer=TfidfVectorizer(use_idf=True, min_df = 0.1 , max_df=0.95, max_features = 235, ngram_range = (1,3))
        #print(tfidf_vectorizer)
        tfidf_vectorizer.fit_transform(training_data[field].values)
        
        names = tfidf_vectorizer.get_feature_names()
        #print(names)
        
        feature_set = tfidf_vectorizer.transform(df[field].values)
        
        train_feature_set=tfidf_vectorizer.transform(training_data[field].values)
        test_feature_set=tfidf_vectorizer.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set,tfidf_vectorizer, feature_set
    
    else:
        #method for Fisher terms
        vectorizer = TfidfVectorizer(use_idf = False,max_df = 0.95, vocabulary=myvocab)
        vectorizer.fit_transform(training_data[field].values)
        #print(X)
        
        feature_set = vectorizer.transform(df[field].values)
        
        train_feature_set= vectorizer.transform(training_data[field].values)
        test_feature_set= vectorizer.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set, vectorizer, feature_set


# In[167]:


def get_top_k_predictions(model,X_test,k):
    
# get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

# GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:,-k:]
    
# GET CATEGORY OF PREDICTIONS
    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
    
# REVERSE CATEGORIES - DESCENDING ORDER OF IMPORTANCE
    preds=[ item[::-1] for item in preds]
    
    return preds


# In[168]:


field = 'processed_abstracts'
feature_rep = 'other'
top_k = 1


# In[169]:


X_train,X_test,feature_transformer, X = extract_features(new_set,field,training_data,testing_data,type=feature_rep)


# In[170]:


print(feature_transformer)


# In[171]:


scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',multi_class = 'auto', random_state=10, C=1,max_iter=1000)
model=scikit_log_reg.fit(X_train,Y_train)


# In[172]:


preds=get_top_k_predictions(model,X_test,top_k)
Y_predicted = ([x[0] for x in preds])
print(Y_predicted)


# In[173]:


cm = confusion_matrix(Y_test,Y_predicted)
print(cm)


# In[174]:


print(sklearn.metrics.classification_report(Y_test,Y_predicted))


# In[175]:


Y = new_set['Label']

scores = cross_val_score(model, X, Y, cv=10)
print(scores)


# In[176]:


avg = scores.mean()
print("Accuracy of the model with a 10-fold cross validation: ", avg * 100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




