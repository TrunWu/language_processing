# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:23:04 2018

@author: Qian
"""

import numpy as np
import pandas as pd
import re
#np.random.seed(13)

from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
import re
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neural_network import MLPClassifier as cnn

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import ImageDataGenerator
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

path = '../../dataset/Tweets-airline-sentiment.csv' 


data = pd.read_csv(path)
text = data['text']
label = data['airline_sentiment']
label_tags = label.unique()


new_text = []
for line in text:
    line = re.sub('[0-9]','', line)
    #line = re.sub('^@\w+ *','', line)  # without [], ^ means match from the start
    line = line.lower()
    new_text.append(re.sub('@\w+ *','', line))    #clean text and get rid of company name
    
new_text = new_text  # testing purpose
new_label = label

tokenizer = Tokenizer()

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

token_text = []
for line in new_text:
    l = tknzr.tokenize(line)
    token_text.append(l)


'''
alltext=' '
for line in new_text:
    # calculate all word vectors and save them in a text file
    alltext = alltext+line

print(len(alltext))
'''
#https://blog.csdn.net/meyh0x5vDTk48P2/article/details/78993600
#https://radimrehurek.com/gensim/models/word2vec.html

model = Word2Vec(token_text, size=200, window=5, min_count=5, workers=4,sg=0)
model.save('word2vec.model')
model = Word2Vec.load('word2vec.model')
#print (model['good'])
#print(model.most_similar('angry'))

#vec = gensim.models.KeyedVectors.load_word2vec_format(token_text, binary=False)
#model.mv.most_similiar(positive=['woman'])



# ================================================================
class WordEmbedding:
    
    def __init__(self, n): #initialize empty, with a dimension size variable
        self.dimensions = n
        self.wordDict = {}
        
    def __init__(self, fileLocation): #initialize from file
        self.wordDict = {}
        with open(fileLocation, encoding="utf-8") as f:
            word_n, self.dimensions = [int(x) for x in f.readline().rstrip().split(" ")]
            for line in f:
                inputWord = line.rstrip().split(" ")
                floatArr = [float(x) for x in inputWord[1:]]
                self.wordDict[inputWord[0]] = np.array(floatArr)
        
    def addWord(self, word, vector): #vector must be Numpy float array
        if len(vector) == self.dimensions and word not in self.wordDict:
            self.wordDict[word] = vector
        else:
            return False #turn into a real error message
        
    def getWordVector(self, word):
        if word in self.wordDict:
            return self.wordDict[word]
        else:
            return False #make a real error message
    
    def cosine_similarity(self, v_1, v_2):
        upper = np.dot(v_1, v_2)
        lower = math.sqrt(np.dot(v_1,v_1)) * math.sqrt(np.dot(v_2,v_2))
        sim = upper / lower
        return sim
    
    def wordSim(self, word1, word2):
        return self.cosine_similarity(self.wordDict[word1],self.wordDict[word2])
    
    #subclass
    class OrderedListTuple:
        def __init__(self, max_size):
            self.content = []
            self.max_size = max_size

        def get (self, LIST, index):
            return LIST[index]
    
        def get_value(self, el):
            return el[1]

        def find_pos (self, element):
            index = 0
            while (index <= len(self.content)-1) and self.get_value(self.get(self.content, index)) > self.get_value(element):
                index += 1
            return index

        def insert_element (self, element):
            pos = self.find_pos (element)
            self.content.insert (pos, element)
            if len(self.content) > self.max_size:
                self.content.pop()
                
    def mostSimilar(self, word, listSize=30):
        outputList = self.OrderedListTuple(listSize)
        v1 = self.wordDict[word]
        for w in self.wordDict:
            if w != word:
                v2 = self.wordDict[w]
                sim = self.cosine_similarity(v1,v2)
                newTuple = (w,sim)
                outputList.insert_element(newTuple)
        return outputList.content
    
    def embedAlgebra(self, w1,w2,w3, n=1):
        searchVector = self.wordDict[w1] + self.wordDict[w2] - self.wordDict[w3]
        
        outputList = self.OrderedListTuple(n)
        for w in self.wordDict:
            v = self.wordDict[w]
            sim = self.cosine_similarity(searchVector,v)
            newTuple = (w,sim)
            outputList.insert_element(newTuple)

        return outputList.content
    
#embeddings = WordEmbedding("vectors_1000.txt")




max_tweet_len = 20

embed_text = []

for t in new_text:
    words = t.split()
    embeds = []
    
    for w in words:
        w = w.casefold()
        #print(w)
        w = w.strip(",.:;_-@#!")
        #print(w)
        if w in model.wv:
            embeds.append(model[w])
            
    vec_embed = np.asarray(embeds)
    
    #print(t)
    
    if vec_embed.shape[0] > max_tweet_len:
        vec_embed = vec_embed[:max_tweet_len, :]
    else:
        #print(vec_embed.shape)
        temp_vec = np.zeros((max_tweet_len, 200))  # output dimention = 200
        
        if vec_embed.shape[0] > 0:
            temp_vec[max_tweet_len - vec_embed.shape[0]:, :] = vec_embed[:,:]
        vec_embed = temp_vec
    
    embed_text.append(vec_embed)

embed_text = np.asarray(embed_text)

print(embed_text.shape)   # this is 3D dimension, need to change to 2d

flat_embeds = np.reshape(embed_text, (embed_text.shape[0], -1))
print(flat_embeds.shape)  # shape in 2d


NB = MultinomialNB()
pc = Perceptron()
svm = LinearSVC()
lr = LogisticRegression()
random_forest  = rf()
KNN = knn(n_neighbors=5)
CNN = cnn()



from sklearn.model_selection import StratifiedKFold as SKF
skf = SKF(n_splits = 5)
X =  flat_embeds
y = label
for clf in [pc, svm, lr, KNN, CNN, random_forest]:
    acc = []
    for train_index, test_index in skf.split(X, y):       
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        acc.append(clf.score(X_test, y_test))
    acc = np.asarray(acc)
    print(clf, acc.mean())




#UniVec = CountVectorizer(max_features = 200, ngram_range = (1,1))
#uni = UniVec.fit_transform(new_text)






