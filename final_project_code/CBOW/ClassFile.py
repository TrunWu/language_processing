# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:20:40 2018

@author: Qian
"""

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
