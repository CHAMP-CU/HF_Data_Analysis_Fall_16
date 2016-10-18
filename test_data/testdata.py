import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loremipsum import get_sentences


dic = pd.read_csv('Data Dictionary - Fall 2016.tsv', delimiter='\t')

nquestions = len(dic)
nresponses = 80

data = pd.DataFrame(columns=dic['Question Text'], index=range(nresponses))
k = 0

for i in range(nquestions):
	if (dic['Data type'][i] == 'Free response') or (dic['Data type'][i] == 'Comment'):
		for j in range(nresponses):
			data.ix[j, i] = get_sentences(1)[0]
	elif dic['Data type'][i] == 'Count':
		for j in range(nresponses):
			data.ix[j, i] = np.random.randint(4)
	elif (dic['Data type'][i] == 'Categorical') or (dic['Data type'][i] == 'Binary'):
		categories = dic['Data values'][i].split(';')
		for j in range(nresponses):
			data.ix[j, i] = categories[np.random.randint(len(categories))]
	elif dic['Data type'][i] == 'Continuous':
		inds = np.array([6, 12, 15, 22, 24, 25, 26])
		k = np.where(i==inds)
		means = np.array([25, 1000, 1000, 100, 67, 16, 12])
		stds = np.array([3, 100, 100, 100, 3, 3, 3])
		for j in range(nresponses):
			data.ix[j, i] = np.floor(np.random.normal(means[k], stds[k]))
	elif dic['Data type'][i] == 'Ordinal':
		mean = np.random.randint(1, 5)
		for j in range(nresponses):
			data.ix[j, i] = np.floor(np.random.normal(mean, 0.3))
	elif dic['Data type'][i] == 'Multiple selection':
		categories = dic['Data values'][i].split(';')
		for j in range(nresponses):
			data.ix[j, i] = categories[np.random.randint(len(categories))]

