# This file contains the functions that are needed for the champ.py script to run
# for the Fall 2016 code.
# Authors: Thomas Jeffries
# Created: 20161110
# Modified: 20161110

def standard_major(given_major):
    major = given_major.lower()
    if given_major == "macs":
        major = "computer science"
    elif given_major == "me":
        major = "mechanical engineering"
    elif given_major == "chemical engneering":
        major = "mechanical engineering"
    return major

def standard_country(given_country):
    country = given_country.lower()
    return country

def standard_languages(given_language):
    language = given_country.lower()
    return country

def gauge_chart_ordinal_cross(responses, categories):
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
	plt.xlabel('%')
	plt.legend(bbox_to_anchor=(0., 1.0, 1.12, -0.25))
	plt.subplots_adjust(left=0.15, right=0.92)
	

def gauge_chart_categorical_cross(responses, categories, values):
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
	plt.xlabel('%')
	plt.legend(bbox_to_anchor=(0., 1.0, 1.12, -0.25))
	plt.subplots_adjust(left=0.15, right=0.92)