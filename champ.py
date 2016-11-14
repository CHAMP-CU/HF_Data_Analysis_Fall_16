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
from champ_funcs import *

#Define plot styles

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='w'
plt.rcParams['figure.facecolor']='w'
plt.rcParams['axes.edgecolor']='w'
plt.rcParams['axes.grid']=False
plt.rcParams['figure.subplot.bottom'] = 0.12
plt.rcParams['savefig.facecolor']='w'
plt.rcParams['axes.color_cycle'] = [u'#30a2da', u'#fc4f30', u'#e5ae38',  '#beaed4', '#fdc086']


#Dictionary data frame
# Load the data dictionary
dic = np.genfromtxt("Data Dictionary - Fall 2016.tsv",
					skip_header=0, names=True, dtype=np.object, delimiter='\t')

# dic['Data_type']
# Save the locations of the different questions for easy access
locs = np.unique(dic['Location'])

# Make the datatype array
datatype = np.zeros(len(dic['Data_type']), np.object)

# Loop through all the different types of data, and change the data type into
# the correct type. Everything was originally saved as an object.
for i in range(len(datatype)):
	if dic['Data_type'][i] == 'Binary':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Categorical':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Continuous':
		datatype[i] = np.float
	elif dic['Data_type'][i] == 'Count':
		datatype[i] = int
	elif dic['Data_type'][i] == 'Date':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Free response':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Comment':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Nominal':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Multiple selection':
		datatype[i] = np.object
	elif dic['Data_type'][i] == 'Ordinal':
		datatype[i] = float
	else:
		datatype[i] = np.object

# Save the file path and name of the data
responses_file = "../CHAMP Test Questionnaire (Responses).xlsx"

#Output path
output_path = '../results'

# Load in the test data responses
#data = np.genfromtxt(responses_file, delimiter='\t',
#					dtype=datatype, missing_values='',
#					skip_header=0, names=True)
#df = DataFrame(data)
df = pd.read_excel(responses_file, names=dic["Fall_2016_Question_Code"], parse_cols=len(dic)-1)
#df = df.convert_objects( dict(zip(dic["Fall_2016_Question_Code"], datatype)))
#There should be no negative values in the data
df[df<0] = np.nan

# Change the column names of the DataFrame
df.columns = dic["Fall_2016_Question_Code"]

#Exclude responses according to dictionary
exclude = np.zeros(df.shape, bool)
for i in range(len(df.T)):
	if dic['Exclude_from'][i]=='':
		continue
	codes = dic['Exclude_from'][i].split(',')
	for code in codes:
		if len(code)==3:
			test = int(code[1:])
			exclude[df.crew_test==test, i] = True
		else:
			re.split("T|CM", code)[1:]
			test = int(code[2])
			crew = int(code[2])
			exclude[(df.crew_test==test)*(df.crew_test==crew), i] = True
df = df.mask(exclude*(datatype!=np.object), try_cast=True)

#Ad-hoc data corrections go here
	
#Classify participant experience
experience = -DataFrame(index=df.index, columns= dic['Data_values'][13].split(';'), 
				dtype=bool)
for i in range(len(df)):
	for j in range(experience.shape[-1]):
		if df.crew_experience[i] != df.crew_experience[i]:
			continue
		experience.ix[i, j] =  experience.columns[j] in df.crew_experience[i]


tag_matrix = (DataFrame(dic).ix[:, -11:] =='1').T
tag_matrix.columns = df.columns
tags = np.array(tag_matrix.index)
tags[3] = 'Location'
tag_matrix.index = tags

# Standardize the majors given
for i in range(0, df.shape[0]):
	df.iloc[i]["crew_major"] = standard_major(df.iloc[i]["crew_major"])
	
	
#Categorize participants
categories = DataFrame(index=np.arange(len(df)))
categories['All'] = np.copy(np.ones(len(df), bool))
categories['Male'] = np.copy(df.crew_gender=='Male')
categories['Female'] = np.copy(df.crew_gender=='Female')
categories['Flight Experience'] = np.copy(df.crew_flight_01=='Yes')
categories['Habitat Experience'] = np.copy(experience.ix[:, [9, 13, 14, 15, 16, 17]].sum(1) > 0)
categories['Space Experience'] = np.copy(experience.ix[:, [7, 8, 9, 10, 11, 12]].sum(1) > 0)
categories['Expert'] = np.copy(experience.sum(1)>=3)
categories['Any Experience'] = np.copy(experience.sum(1)>0)
categories['No Experience'] = np.copy(experience.sum(1)==0)
categories['US National'] = df.crew_national.str.count("United|America|USA|US|United States|U.S.") > 0
categories['International'] = -categories['US National']
categories['30 and older'] = np.copy(df.crew_age>= 30)
categories['Under 30'] = np.copy(df.crew_age< 30) 
ansur_f = np.array([58.5, 14.9, 25.6])
ansur_m = np.array([76.6, 22.1, 35.8])
above = (df.crew_height > ansur_m[0])+(df.crew_shoulder > ansur_m[1])+(df.crew_thumb > ansur_m[2])
below = (df.crew_height < ansur_f[0])+(df.crew_shoulder < ansur_f[1])+(df.crew_thumb < ansur_f[2])
#categories['Above Limits'] = np.copy(above)
categories['First Quartile'] = df.crew_height <= 63.
categories['Second Quartile'] = (df.crew_height > 63.)*(df.crew_height <= 67.)
categories['Third Quartile'] = (df.crew_height > 67.)*(df.crew_height <= 71.)
categories['Fourth Quartile'] = (df.crew_height > 71.)

#categories['Below Limits'] = np.copy(below)
categories['New Participant'] = np.copy(df.crew_prior=='No')
categories['Repeat Participant'] = np.copy(df.crew_prior=='Yes, in Spring 2016')
categories['CHAMP'] = np.copy((df.crew_champ=='Yes (former)')+(df.crew_champ=='Yes (current)'))
categories['Non-CHAMP'] = np.copy(df.crew_champ=='No')
categories['CM1'] = np.copy(df.crew_id==1)
categories['CM2'] = np.copy(df.crew_id==2)
categories['CM3'] = np.copy(df.crew_id==3)
categories['CM4'] = np.copy(df.crew_id==4)

category_names = categories.columns

#Category classes
subcat_height = pd.concat([categories['All'], 
							categories['First Quartile'],  
							categories['Second Quartile'],
							categories['Third Quartile'],
							categories['Fourth Quartile']], axis=1)
subcat_gender = pd.concat([categories['All'], 
							categories['Male'],  categories['Female']], axis=1)
subcat_champ = pd.concat([categories['All'], 
							categories['CHAMP'],  categories['Non-CHAMP']], axis=1)
subcat_repeat = pd.concat([categories['All'], categories['Repeat Participant'],  
							categories['New Participant']], axis=1)
subcat_cm = pd.concat([categories['All'], categories['CM1'], categories['CM2'], 
							categories['CM3'], categories['CM4']], axis=1)
subcat_national= pd.concat([categories['All'], categories['US National'], 
							categories['International']], axis=1)
subcat_experience = pd.concat([categories['All'], categories['Flight Experience'], 
								categories['Habitat Experience'], 
								categories['Space Experience'],
								categories['Expert'], 
								categories['Any Experience'],
								categories['No Experience']], axis=1)
								
subcats = [subcat_height, subcat_gender, subcat_experience, subcat_champ, subcat_repeat, 
				subcat_cm, subcat_national]
subcat_names = ['height', 'gender', 'experience', 'champ', 'repeat', 'cm', 'national']


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
mask = (dic['Data_type']=="Ordinal")+(dic['Data_type']=="Categorical")
mask+= (dic['Data_type']=="Binary")+(dic['Data_type']=="Count")
mask+= (dic['Data_type']=="Free Response")+(dic['Data_type']=="Comment")
mask+= (dic['Data_type']=="Free response")
mask+= (dic['Data_type']=="Multiple selection")+(dic['Data_type']=="Multiple Selection")

#Exclude questions with fewer than 8 responses
mask *= (df.count(axis=0) > 8)#+(df.sum(0).str.count('NaN') < len(df)-8)
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
			#total = (subframe.ix[:, i].ix[categories.ix[:, j]]!='NaN').sum()
			#total_not = (subframe.ix[:, i].ix[-categories.ix[:, j]]!='NaN').sum()
			total = subframe.ix[:, i].ix[categories.ix[:, j]].valid().count()
			total_not = subframe.ix[:, i].ix[categories.ix[:, j]].valid().count()
			if False:
				for k in range(len(responsetypes)):
					width[k] = (responsetypes[k]==subframe.ix[:, i].ix[categories.ix[:, j]].astype(str)).sum()*1./total
					width_not[k] = (responsetypes[k]==subframe.ix[:, i].ix[-categories.ix[:, j]].astype(str)).sum()*1./total_not
					
				pval = chi2_contingency(np.vstack((width*total, width_not*total_not))[:, (width_not!=0)])[1]
			else:
				for k in range(len(responsetypes)):
					width[k] = (responsetypes[k]==subframe.ix[:, i].ix[categories.ix[:, j]].astype(str)).sum()*1./total
				pval = np.nan
			
			print '%21s\t%i' % (category_names[j], total) + '\t%3.1f%%'*len(responsetypes) % tuple(width*100)+'\t%3.2f' % (pval)+'*'*(pval < 0.05)
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
				cat_tags = '%s, '*(categories.ix[j].sum()-1) % tuple(category_names[categories.ix[j]][1:])
				print '\n'.join(wrap(cat_tags, 70))
				print '\t'+'\n\t'.join(wrap(subframe.ix[:, i][j], 70))
	elif (dic['Data_type'][mask][i].lower() == 'multiple selection'):
		selections = -DataFrame(index=df.index, columns= dic['Data_values'][mask][i].split(';'), 
				dtype=bool)
		for j in range(len(df)):
			for k in range(selections.shape[-1]):
				selections.ix[j, k] =  selections.columns[k] in [subframe.ix[:, i][j]]
		for j in range(len(selections.columns)):
			print '%61s\t' % (selections.columns[j]) + '\t%2.1f%%' % (100*selections.mean(0)[j])



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

makefigs = True
if makefigs:
	for i in np.argsort(np.array(dic['Order_Asked'][mask], int)):
		if dic['Data_type'][mask][i] == 'Ordinal':
			gauge_chart_ordinal_cross(subframe.ix[:, i], categories)
			plt.savefig(output_path+"/figs/all/"+'%03i' % i)
			plt.close()
			gauge_chart_ordinal_cross(subframe.ix[:, i], DataFrame(categories.All))
			plt.savefig(output_path+"/figs/all/"+'%03i' % i)
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
			plt.savefig(output_path+"/figs/all/"+'%03i' % i)
			plt.close()
			for j in range(len(subcats)):
				gauge_chart_categorical_cross(subframe.ix[:, i], subcats[j])
				plt.savefig(output_path+"/figs/"+subcat_names[j]+'/'+'%03i' % i)
				plt.close()



sys.exit()