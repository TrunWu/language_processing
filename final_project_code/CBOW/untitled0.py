# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 21:53:45 2018

@author: Qian
"""

from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('word2vec_300.bin', binary=True)
model.save_word2vec_format('word2vec_300.txt', binary=False)