
# coding: utf-8

# In[1]:


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


# In[2]:


path = '../../dataset/Tweets-airline-sentiment.csv'   # ../ = upper directory

print (u'\U00002764')
print (u'\U0001f44d')
print (u'\U0001f604')
print (u'\U0001f601')
print (u'\U0001f621')


# In[3]:


data = pd.read_csv(path)
text = data['text']
label = data['airline_sentiment']
label_tags = label.unique()

#replace text label with one-hot-labels
#new_label= []
#for l in label:
#    if l == label_tags[0]:
#        new_label.append(np.array([0,0,1]))
#    elif l == label_tags[1]:
#        new_label.append(np.array([0,1,0]))
#    else:
#        new_label.append(np.array([1,0,0]))
# above is one-hot-labels, represent labels in matrics, but can not be divided by stratifield kfold

new_text = []
for line in text:
    line = re.sub('[0-9]','', line)
    #line = re.sub('^@\w+ *','', line)  # without [], ^ means match from the start
    line = line.lower()
    new_text.append(re.sub('@\w+ *','', line))    #clean text and get rid of company name
    
new_text = new_text  # testing purpose
new_label = label
new_text[20:50]


# In[4]:


new_text[20:30]


# In[5]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(new_text)
corpus = tokenizer.texts_to_sequences(new_text)  #Transforms each text in texts in a sequence of integers.
#Only top "num_words" most frequent words will be taken into account.
#Only words known by the tokenizer will be taken into account.
# help(Tokenizer.texts_to_sequences)

#corpus = new_text  # Qian added this line

nb_samples = sum(len(s) for s in corpus)    # each s is a word(integer), this is just counting total words in corpus

# The training phase is by means of the fit_on_texts method and you can see the word index using the word_index property
# http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/
V = len(tokenizer.word_index) + 1    
# help(Tokenizer)

dim = 100   # for output dimension
window_size = 3   # this is create window for CBOW, so it takes order into consideration


# In[6]:


#help(Tokenizer)
#tokenizer.word_index


# In[7]:


# corpus   # for each line in twitter, it generate vectors or some kind number for each word
# help(Tokenizer)
# tokenizer.word_index
# help(sequence.pad_sequences)


# In[8]:


def generate_data(corpus, window_size, V):
    maxlen = window_size*2  # left 2 + right 2 so max 4 words
    for words in corpus:
        L = len(words)    # how many words are in each line in the corpus. 'words' here means 'line'
        #print(words,L) 
        for index, word in enumerate(words):
            contexts = []
            labels   = []            
            s = index - window_size   # start
            e = index + window_size + 1   # end
            # the window size works in a way: it only looks at the left (window size) words and the right (window size) words.
            # here is looks at the left 2 and the right 2 words of a selected index word.
            
            contexts.append([words[i] for i in range(s, e) if 0 <= i < L and i != index])
            # above: it looks at all the words from start to end in the selected range, and without look at the index word 
            
            labels.append(word)
            
            x = sequence.pad_sequences(contexts, maxlen=maxlen) # pad sequence to the same length. add 0 to short sentences
            # all sentences has to be the same length otherwise it won't work. 
              
            y = np_utils.to_categorical(labels, V)
            yield (x, y)


# In[9]:


cbow = Sequential()  # create a sequence of actions below
cbow.add(Embedding(input_dim=V, output_dim=dim, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim,)))
cbow.add(Dense(V, activation='softmax'))


# In[ ]:


print('start')
cbow.compile(loss='categorical_crossentropy', optimizer='adadelta')
print('finished')


# In[ ]:


for ite in range(10):
    loss = 0.
    for x, y in generate_data(corpus, window_size, V):
        loss += cbow.train_on_batch(x, y)
    print('finished', ite)
    print(ite, loss)


# In[ ]:


f = open('vectors.txt' ,'w', encoding = 'utf8')
f.write('{} {}\n'.format(V-1, dim))


# In[ ]:


vectors = cbow.get_weights()[0]
for word, i in tokenizer.word_index.items():
    # calculate all word vectors and save them in a text file
    str_vec = ' '.join(map(str, list(vectors[i, :])))
    f.write('{} {}\n'.format(word, str_vec))
f.close()


# In[ ]:


w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)


# In[ ]:


w2v.most_similar(positive=['good'])


# In[ ]:


w2v.most_similar(positive=['bad'])


# In[ ]:


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
    
embeddings = WordEmbedding("vectors_1000.txt")


# In[ ]:


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
        if w in embeddings.wordDict:
            embeds.append(embeddings.getWordVector(w))
            
    vec_embed = np.asarray(embeds)
    
    #print(t)
    
    if vec_embed.shape[0] > max_tweet_len:
        vec_embed = vec_embed[:max_tweet_len, :]
    else:
        #print(vec_embed.shape)
        temp_vec = np.zeros((max_tweet_len, 100))  # output dimention = 100
        
        if vec_embed.shape[0] > 0:
            temp_vec[max_tweet_len - vec_embed.shape[0]:, :] = vec_embed[:,:]
        vec_embed = temp_vec
    
    embed_text.append(vec_embed)

embed_text = np.asarray(embed_text)

print(embed_text.shape)   # this is 3D dimension, need to change to 2d


# In[ ]:


flat_embeds = np.reshape(embed_text, (embed_text.shape[0], -1))
print(flat_embeds.shape)  # shape in 2d


# In[ ]:


NB = MultinomialNB()
pc = Perceptron()
svm = LinearSVC()
lr = LogisticRegression()
random_forest  = rf()
KNN = knn(n_neighbors=5)
CNN = cnn()


# In[ ]:


from sklearn.model_selection import StratifiedKFold as SKF
skf = SKF(n_splits = 5)
X =  flat_embeds
y = label[:100]
for clf in [pc, svm, lr, KNN, CNN, random_forest]:
    acc = []
    for train_index, test_index in skf.split(X, y):       
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        acc.append(clf.score(X_test, y_test))
    acc = np.asarray(acc)
    print(clf, acc.mean())


# In[ ]:


UniVec = CountVectorizer(max_features = 100, ngram_range = (1,1))
uni = UniVec.fit_transform(new_text)

