#dataset preprocessing file
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re


def read_data(data_path, feature = 'Unigram', max_feature_num = 500):
#feature: the text feature, could be 'Unigram, Bigram, Trigram or Mixing Unigram with Bigram'
	data = pd.read_csv(data_path)
	text = data['text']
	label = data['airline_sentiment']
	label_tags = label.unique()
	#replace text label with one-hot-labels
	new_label= []
	for l in label:
		if l == label_tags[0]:
			new_label.append(np.array([0,0,1]))
		elif l == label_tags[1]:
			new_label.append(np.array([0,1,0]))
		else:
			new_label.append(np.array([1,0,0]))
	#get rid of '@airline_company_name
	new_text = []
	for t in text:
		new_text.append(re.sub('^@\\w+ *','', t))
	if feature == 'Unigram':
		Vec = CountVectorizer(max_features = max_feature_num, ngram_range=(1,1))
		out = Vec.fit_transform(new_text)
	elif feature == 'Bigram':
		Vec = CountVectorizer(max_features = max_feature_num, ngram_range=(2,2))
		out = Vec.fit_transform(new_text)
	elif feature == 'Trigram':
		Vec = CountVectorizer(max_features= max_feature_num, ngram_range=(3,3))
		out = Vec.fit_transform(new_text)
	else:
	# mix bigram and unigram
		Vec = CountVectorizer(max_features = max_feature_num, ngram_range = (1,2))
		out = Vec.fit_transform(new_text)
	return out, new_label