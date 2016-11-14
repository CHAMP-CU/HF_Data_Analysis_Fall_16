import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from pandas import DataFrame
import pandas as pd
import matplotlib
import json
from textwrap import wrap
from scipy import stats
import re
import sys
import champ_funcs as cf

# Load the data
from load_CHAMP import *

#Define plot styles
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='w'
plt.rcParams['figure.facecolor']='w'
plt.rcParams['axes.edgecolor']='w'
plt.rcParams['axes.grid']=False
plt.rcParams['figure.subplot.bottom'] = 0.12
plt.rcParams['savefig.facecolor']='w'
plt.rcParams['axes.color_cycle'] = [u'#30a2da', u'#fc4f30', u'#e5ae38',  '#beaed4', '#fdc086']

#What locations can be smaller?
smaller = (df.ix[:, dic['Aspect']=='smaller']=='Yes')
smaller.columns = ['Airlock', 'Galley', 'Sleep', 'Storage', 'Command', 'Science', 'ECLSS',
					'Emergency Path', 'Technology Development', 'Exercise', 'Hygiene']
(100*smaller.mean(0))[np.argsort(smaller.mean(0))].plot(kind='barh', stacked=True)
plt.subplots_adjust(left=0.3)
plt.xlabel('% Yes')
plt.title('\n'.join(wrap("Could the volume of the following spaces be smaller "+
							"and still acceptable for the tasks you performed?", 70)),
							size='small')



### This part was removed because we don't have a seperate free response form
#"free responses" data frame
#df = DataFrame(data)
#responses = np.genfromtxt("Questionnaire Data - Free responses.tsv",
#					skip_header=1, names=True, dtype=np.object, delimiter='\t')
#rf = DataFrame(responses)

#Create tables
'''mask = (dic['Data_type']=="Ordinal")+(dic['Data_type']=="Categorical")
mask+= (dic['Data_type']=="Binary")+(dic['Data_type']=="Count")
mask+= (dic['Data_type']=="Free Response")+(dic['Data_type']=="Comment")
mask+= (dic['Data_type']=="Free response")
mask+= (dic['Data_type']=="Multiple selection")+(dic['Data_type']=="Multiple Selection")
mask+= (dic['Data_type']=="Continuous")'''

#Exclude questions with fewer than 8 responses
mask = np.array(df.count(axis=0) > 8)#+(df.sum(0).str.count('NaN') < len(df)-8)
subframe = df.ix[:, mask]
for i in np.argsort(np.array(dic['Order_Asked'][mask], int))[1:]:
	print np.array(dic['Order_Asked'][mask], int)[i]
	print dic['Location'][mask][i]
	print dic['Question_Text'][mask][i]
	tags = tag_matrix.ix[:, mask].ix[:, i]
	print "Tags: "+'%s, '*tags.sum() % tuple(tags.index.str.lower()[tags])
	if dic['Data_type'][mask][i] == 'Ordinal':
		print dic['Data_values'][mask][i]
		print "Category\t\tn\tMean\t1\t2\t3\t4\t5\t6\t(4-6)\tp-value"
		for j in range(len(category_names)):
			width = np.zeros(6)
			total = subframe.ix[:, i].ix[categories.ix[:, j]].valid().count()
			for k in range(6):
				width[k] = (subframe.ix[:, i].ix[categories.ix[:, j]]==(k+1)).mean()*1.
			pval = stats.ranksums(subframe.ix[:, i].ix[categories.ix[:, j]].dropna(), subframe.ix[:, i].ix[-categories.ix[:, j]].dropna())[1]
			print '%21s\t%i' % (category_names[j], total) + '\t%2.1f' % (subframe.ix[:, i].ix[categories.ix[:, j]]).mean() + '\t%3.1f%%'*6 % tuple(width*100)+'\t%3.1f%%' % (width[3:].sum()*100) +'\t%3.2f' % (pval)+'*'*(pval < 0.05)
		print
	elif dic['Data_type'][mask][i] == 'Binary':
		responsetypes = dic['Data_values'][mask][i].split(';')
		print "Category\t\t n\t"+ '%s\t'*len(responsetypes) % tuple(responsetypes)+"p-value"
		for j in range(len(category_names)):
			yes = (subframe.ix[:, i].ix[categories.ix[:, j]]==responsetypes[0]).sum()
			no = (subframe.ix[:, i].ix[categories.ix[:, j]]==responsetypes[1]).sum()
			total = (yes+no)*1.
			if category_names[j] !='All':
				p0 = subframe.ix[:, i].ix[-categories.ix[:, j]].str.contains(responsetypes[0]).mean()
				pval = stats.binom_test((yes, no), p=p0)
			else:
				pval = np.nan
			print '%21s\t%i' % (category_names[j], total) + '\t%3.1f%%'*2 % (yes*1./total*100, no*1./total*100)+'\t%3.2f' % (pval)+'*'*(pval < 0.05)
		print
	elif (dic['Data_type'][mask][i] == 'Categorical') *(i>2):
		responsetypes = dic['Data_values'][mask][i].split(';')
		print "Category\t\t n\t"+ '%s\t'*len(responsetypes) % tuple(responsetypes)+"p-value"
		for j in range(len(category_names)):
			width = np.zeros(len(responsetypes))
			width_not = np.zeros(len(responsetypes))
			total = subframe.ix[:, i].ix[categories.ix[:, j]].valid().count()
			total_not = subframe.ix[:, i].ix[-categories.ix[:, j]].valid().count()
			for k in range(len(responsetypes)):
				width[k] = (responsetypes[k]==subframe.ix[:, i].ix[categories.ix[:, j]].astype(str)).sum()*1.
				width_not[k] = (responsetypes[k]==subframe.ix[:, i].ix[-categories.ix[:, j]].astype(str)).sum()*1.
			obs = np.vstack((width, width_not))
			if obs[1].min() > 0:
				pval = stats.chi2_contingency(obs[:, obs[1]>0])[1]
			else:
				pval = np.nan

			print '%21s\t%i' % (category_names[j], total) + '\t%3.1f%%'*len(responsetypes) % tuple(width/total*100)+'\t%3.2f' % (pval)+'*'*(pval < 0.05)
		print
	elif (dic['Data_type'][mask][i] == 'Count')*(i > 5):

		responsetypes = np.array(np.unique(subframe.ix[:, i].dropna()), int)
		print "Category\t\tn\t"+"Mean"+ "\t%s"*len(responsetypes) % tuple(responsetypes)+ "\tp-value"
		for j in range(len(category_names)):
			width = np.zeros(len(responsetypes))
			total = np.in1d(subframe.ix[:, i].ix[categories.ix[:, j]], responsetypes).sum()
			for k in range(len(responsetypes)):
				width[k] = (subframe.ix[:, i].ix[categories.ix[:, j]]==responsetypes[k]).sum()*1./total
			pval = stats.ranksums(subframe.ix[:, i].ix[categories.ix[:, j]].dropna(), subframe.ix[:, i].ix[-categories.ix[:, j]].dropna())[1]
			print '%21s\t%i' % (category_names[j], total) + '\t%2.1f' % (subframe.ix[:, i].ix[categories.ix[:, j]]).mean() + '\t%3.1f%%'*len(responsetypes) % tuple(width*100)+'\t%3.2f' % (pval)+'*'*(pval < 0.05)
		print
	elif (dic['Data_type'][mask][i].lower() == 'free response') or (dic['Data_type'][mask][i] == 'Comment'):
		for j in range(len(subframe.ix[:, i])):
			if (subframe.ix[:, i][j] !='NaN') and (subframe.ix[:, i][j]== subframe.ix[:, i][j]):
				print
				cat_tags = '\t\t'+'%s, '*(categories.ix[j].sum()-1) % tuple(category_names[categories.ix[j]][1:])

				print '\t'+'\n\t'.join(wrap(subframe.ix[:, i][j], 70))
				print '\n\t\t'.join(wrap(cat_tags, 70))
	elif (dic['Data_type'][mask][i].lower() == 'multiple selection'):
		selections = -DataFrame(index=df.index, columns= dic['Data_values'][mask][i].split(';'),
				dtype=bool)
		for j in np.where(subframe.ix[:, i]==subframe.ix[:, i])[0]:
			for k in range(selections.shape[-1]):
				selections.ix[j, k] =  selections.columns[k] in subframe.ix[:, i][j]

		for j in range(len(selections.columns)):
			print '%70s\t' % (selections.columns[j]) + '\t%2.1f%%' % (100*selections.mean(0)[j])
	elif dic['Data_type'][mask][i] == 'Continuous':
		print dic['Data_values'][mask][i]
		print "Category\t\tn\tMean\tStd.\tMin.\t25%\t50%\t75%\tMax.\tp-value"
		for j in range(len(category_names)):
			description = subframe.ix[:, i].ix[categories.ix[:, j]].describe()
			pop1 = subframe.ix[:, i].ix[categories.ix[:, j]].dropna()
			pop2 = subframe.ix[:, i].ix[-categories.ix[:, j]].dropna()
			pval = stats.ttest_ind(pop1, pop2)[1]
			print '%21s\t%i' % (category_names[j], description['count'])+'\t%3.1f'*(len(description)-1) % tuple(description[1:])+ '\t%3.2f' % (pval)+'*'*(pval < 0.05)
		print

def gauge_chart_ordinal_cross(responses, categories):
	values = dic['Data_values'][responses.name==dic['Fall_2016_Question_Code']][0]
	stack = DataFrame(columns=np.arange(1, 7), index=categories.columns)
	for i in range(categories.shape[-1]):
		stack.ix[i] = np.histogram(responses[categories.ix[:, i]],
			np.arange(1, 8), normed=True)[0]
	(100*stack[::-1].ix[:,  ::-1]).plot(kind='barh', stacked=True,
		width=1, edgecolor='w', colors=plt.cm.RdBu_r(np.linspace(0.25, 0.75, 6)),
		align='edge', figsize=(12, 6), legend=False)
	plt.title("\n".join(wrap(dic['Question_Text']\
			[responses.name==dic['Fall_2016_Question_Code']][0], 88)), size='medium')
	for i in range(6):
		plt.axvline(np.cumsum(np.histogram(responses[categories.ix[:, 0]],
		np.arange(1, 8), normed=True)[0][::-1])[i]*100, color='lightgray')
	plt.axis('tight')
	plt.xticks(np.arange(0, 101, 10), ('%i%% '*11 % tuple(np.arange(0, 101, 10))).split())
	#plt.legend(bbox_to_anchor=(0., 1.0, 1.12, -0.25))
	plt.legend(bbox_to_anchor=(0.3, -0.04, 0.5, 0), ncol=6, fontsize='medium', framealpha=0)
	plt.annotate(values.split(';')[1].split('-')[1], (0.28, 0.05), xycoords='figure fraction', ha='right', va='center')
	plt.annotate(values.split(';')[0].split('-')[1], (0.8, 0.05), xycoords='figure fraction', ha='left', va='center')
	plt.subplots_adjust(left=0.18)


def gauge_chart_categorical_cross(responses, categories):
	values = dic['Data_values'][responses.name==dic['Fall_2016_Question_Code']][0].split(';')
	stack = DataFrame(columns=values, index=categories.columns)
	for i in range(len(stack)):
		for j in range(len(stack.T)):
			stack.ix[i, j] = (np.array(responses[categories.ix[:, i]], str)==values[j]).sum()
	(100*stack.T/stack.sum(1)).T[::-1].plot(kind='barh', stacked=True, width=1,
		edgecolor='w', legend=False, align='edge', figsize=(12, 6))
	plt.title("\n".join(wrap(dic['Question_Text']\
			[responses.name==dic['Fall_2016_Question_Code']][0], 88)), size='medium')
	for i in range(len(stack.T)):
		plt.axvline(np.cumsum((stack.T/stack.sum(1)).T.ix[0])[::-1][i]*100, color='lightgray')
	plt.axis('tight')
	plt.xticks(np.arange(0, 101, 10), ('%i%% '*11 % tuple(np.arange(0, 101, 10))).split())
	plt.legend(bbox_to_anchor=(0.55, -0.05, 0.5, 0), ncol=len(values), fontsize='small')
	plt.subplots_adjust(left=0.18, right=0.92)


def gauge_chart_categorical_cross(responses, categories):
	values = dic['Data_values'][responses.name==dic['Fall_2016_Question_Code']][0].split(';')
	stack = DataFrame(columns=values, index=categories.columns)
	for i in range(len(stack)):
		for j in range(len(stack.T)):
			stack.ix[i, j] = (np.array(responses[categories.ix[:, i]], str)==values[j]).sum()
	(100*stack.T/stack.sum(1)).T[::-1].plot(kind='barh', stacked=True, width=1,
		edgecolor='w', legend=False, align='edge', figsize=(12, 6))
	plt.title("\n".join(wrap(dic['Question_Text']\
			[responses.name==dic['Fall_2016_Question_Code']][0], 88)), size='medium')
	for i in range(len(stack.T)):
		plt.axvline(np.cumsum((stack.T/stack.sum(1)).T.ix[0])[::-1][i]*100, color='lightgray')
	plt.axis('tight')
	plt.xticks(np.arange(0, 101, 10), ('%i%% '*11 % tuple(np.arange(0, 101, 10))).split())
	plt.legend(bbox_to_anchor=(0.55, -0.05, 0.5, 0), ncol=len(values), fontsize='small')
	plt.subplots_adjust(left=0.18, right=0.92)


makefigs = True
if makefigs:
	for i in np.argsort(np.array(dic['Order_Asked'][mask], int)):
		if dic['Data_type'][mask][i] == 'Ordinal':
			gauge_chart_ordinal_cross(subframe.ix[:, i], categories)
			plt.savefig(output_path+"/figs/all/"+'%03i' % i)
			plt.close()
			gauge_chart_ordinal_cross(subframe.ix[:, i], DataFrame(categories.All))
			plt.savefig(output_path+"/figs/topline/"+'%03i' % i)
			plt.close()
			for j in range(len(subcats)):
				gauge_chart_ordinal_cross(subframe.ix[:, i], subcats[j])
				plt.savefig(output_path+"/figs/"+subcat_names[j]+'/'+'%03i' % i)
				plt.close()
		elif (dic['Data_type'][mask][i] == 'Categorical')+(dic['Data_type'][mask][i] == 'Binary'):
			gauge_chart_categorical_cross(subframe.ix[:, i], categories)
			plt.savefig(output_path+"/figs/all/"+'%03i' % i)
			plt.close()
			gauge_chart_categorical_cross(subframe.ix[:, i], DataFrame(categories.All))
			plt.savefig(output_path+"/figs/topline/"+'%03i' % i)
			plt.close()
			for j in range(len(subcats)):
				gauge_chart_categorical_cross(subframe.ix[:, i], subcats[j])
				plt.savefig(output_path+"/figs/"+subcat_names[j]+'/'+'%03i' % i)
				plt.close()
		elif (dic['Data_type'][mask][i] == 'Categorical')+(dic['Data_type'][mask][i] == 'Binary'):
			gauge_chart_categorical_cross(subframe.ix[:, i], categories)
			plt.savefig(output_path+"/figs/all/"+'%03i' % i)
			plt.close()
			gauge_chart_categorical_cross(subframe.ix[:, i], DataFrame(categories.All))
			plt.savefig(output_path+"/figs/topline/"+'%03i' % i)
			plt.close()
			for j in range(len(subcats)):
				gauge_chart_categorical_cross(subframe.ix[:, i], subcats[j])
				plt.savefig(output_path+"/figs/"+subcat_names[j]+'/'+'%03i' % i)
				plt.close()


sys.exit()