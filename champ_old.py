import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from pandas import DataFrame
import matplotlib
import json
from textwrap import wrap
from scipy import stats
import re

#Dictionary data frame
dic = np.genfromtxt("Questionnaire Data - Dictionary.tsv",
					skip_header=1, names=True, dtype=np.object, delimiter='\t')

dic['Type_of_Data']
locs = np.unique(dic['Location'])
datatype = np.zeros(len(dic['Type_of_Data']), np.object)
for i in range(len(datatype)):
	if dic['Type_of_Data'][i] == 'Binary':
		datatype[i] = np.object
	elif dic['Type_of_Data'][i] == 'Categorical':
		datatype[i] = np.object
	elif dic['Type_of_Data'][i] == 'Continuous':
		datatype[i] = np.float
	elif dic['Type_of_Data'][i] == 'Count':
		datatype[i] = int
	elif dic['Type_of_Data'][i] == 'Date':
		datatype[i] = np.object
	elif dic['Type_of_Data'][i] == 'Free-response':
		datatype[i] = np.object
	elif dic['Type_of_Data'][i] == 'Nominal':
		datatype[i] = np.object
	elif dic['Type_of_Data'][i] == 'Ordinal':
		datatype[i] = int
	else:
		datatype[i] = np.object

#"data" data frame
data = np.genfromtxt("Questionnaire Data - Data.tsv",
					skip_header=1, names=True, delimiter='\t',  dtype=tuple(datatype),
					missing_values=(-999, -888, '-999', '-888'), skip_footer=1)

#"free responses" data frame
df = DataFrame(data)
responses = np.genfromtxt("Questionnaire Data - Free responses.tsv",
					skip_header=1, names=True, dtype=np.object, delimiter='\t')
rf = DataFrame(responses)

#Plot binary responses

binary = (dic['Type_of_Data']=='Categorical')+(dic['Type_of_Data']=='Binary')

plt.figure(figsize=(8, 8))
#Crew Experience
plt.subplot(311)
labels = ['Aircraft', 'Spacecraft', 'Long-duration spaceflight', 'Spacecraft simulators', 'Human spacecraft']
plt.title('Flight Experience')
subframe = df[df >0].ix[:, binary].ix[:, 1:6]
(100*(subframe=='y').sum()*1./((subframe=='y').sum()+(subframe=='n').sum())).plot('bar')
plt.xticks(np.arange(5), labels, rotation=0, size='xx-small')
plt.ylabel("%")
for i in range(5):
	plt.annotate('%3.1f%%' % (100*(subframe=='y').sum()*1./((subframe=='y').sum()+(subframe=='n').sum()))[i], (i, 10), ha='center')

plt.subplot(313)
labels = ['Table Collapsed?', 'Private Communications?', 'Egress Training?']
plt.title('Overall')
subframe = df[df >0].ix[:, binary].ix[:, 8:-3]
(100*((subframe=='y').sum()*1./((subframe=='y').sum()+(subframe=='n').sum()))).plot('barh')
plt.subplots_adjust(left=0.2)
plt.yticks(size='xx-small')
plt.xlabel("% 'yes'")

for i in range(3):
	plt.annotate('%3.1f%%' % (100*((subframe=='y').sum()*1./((subframe=='y').sum()+(subframe=='n').sum())))[i], (10, i), va='center')


plt.subplot(312)
labels = ['Table Collapsed?', 'Private Communications?']
plt.title('Preferences')
subframe = df[df >0].ix[:, binary].ix[:, 6:8]
((100*((subframe=='Available')+(subframe=='Private')).sum())/(((subframe=='Collpased')+(subframe=='Private')+(subframe=='Communal')+(subframe=='Available')).sum())).plot('bar')
plt.xticks(np.arange(2), labels, rotation=0, size='x-small')
plt.ylabel("% 'yes'")
for i in range(2):
	plt.annotate('%3.1f%%' % ((100*((subframe=='Available')+(subframe=='Private')).sum())/(((subframe=='Collpased')+(subframe=='Private')+(subframe=='Communal')+(subframe=='Available')).sum()))[i], (i, 10), ha='center')
plt.savefig('binary')

#Free-response evaluation
responsecount = 0
sentiment = np.array([], float)
sentimentloc = np.array([], str)


f = open("responses.txt", 'w')
responsestr = ''
for i in range(len(dic['Question'])):
    question = dic['Code'][i]
    #print dic['Question'][i]
    responsestr += dic['Question'][i]+'\t\n'
    for j in range(len(responses[question])):
		item = responses[question][j]
		if (item != '') * (item != '-888') * (item!='-999') * (len(item) > 3):
			#print '\t'+item
			#print '\t\t'+str(TextBlob(item).sentiment)
			sentiment = np.append(sentiment, TextBlob(item).sentiment.polarity)
			sentimentloc = np.append(sentimentloc, dic['Location'][i])
			responsecount+=1
			responsestr += '\t(%s%s) ' % (df.ix[j, i], ', ASTRONAUT'*(df.crew_experience_space_[j]=='y')) +item+' \n'
f.write(responsestr)
f.close()

sf = DataFrame(np.zeros(df.shape), columns=dic['Code'])
pf = DataFrame(np.zeros(df.shape), columns=dic['Code'])

sf[rf < 0] = np.nan
pf[rf < 0] = np.nan
for i in range(4, len(sf.T)):
	for j in range(len(sf)):
		sf.set_value(j, dic['Code'][i], TextBlob(rf.ix[j, i]).sentiment.polarity)
		pf.set_value(j, dic['Code'][i], TextBlob(rf.ix[j, i]).sentiment.subjectivity)

#Free-response polarity boxplots

plt.figure()
i = 0
for location in locs:
    plt.boxplot(sentiment[sentimentloc==location], positions=np.array([i]))
    i+=1
plt.xticks(np.arange(len(locs)), locs, rotation=90, ha='center')
plt.axis('tight')
plt.ylabel("Polarity")
plt.savefig("figs/polarity_box")

#Average Crew Response
egress = np.genfromtxt("Emergency Egress Times - Sheet1.tsv", names=True, delimiter='\t',
			dtype=(np.object, int, np.object, float, int, int, np.object))
egress_mask = np.ones(17, bool)
egress_mask[[0, 13]] = False
crew = DataFrame()
for i in range(17):
    crew[i] = df[df >0][df.crew_test_number_==i].mean()
crew = crew.T
H = np.matrix([]).T

#Ordinal response boxplots


with plt.style.context('fivethirtyeight'):
	plt.figure()
	df[df > 0].ix[:, dic['Type_of_Data']=="Ordinal"].plot(kind='box', vert=False, patch_artist=True)
	plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
	plt.subplots_adjust(right=0.7)
	plt.yticks(size='xx-small', va='bottom')
	plt.savefig("figs/ordinal_box")

def pheightf(h):
	return 1/np.sqrt(2*np.pi*2.0)*np.exp(-(h-63.8)**2/2.0**2)

def pheightm(h):
	return 1/np.sqrt(2*np.pi*3.1)*np.exp(-(h-69.8)**2/3.1**2)

genderapi= False
prior = np.zeros((df.crew_id_name_!='').sum())+0.5
prior0 = 0.5
posterior = np.zeros((df.crew_id_name_!='').sum()) #Probability a participant is male given first name and height
out= []
firstname = np.zeros((df.crew_id_name_!='').sum(), np.object)
for i in range(len(firstname)):
	firstname[i] = df.crew_id_name_[i].split()[0]
if genderapi:
	from genderize import Genderize
	for i in range(len(posterior)/10+1):
		plt.pause(3)
		out =  np.append(out, Genderize().get(firstname[i*10:(i+1)*10]))
	np.savez('genderize', genders=out)

out = np.load('genderize.npz')['genders']#[:-1]
for i in range(len(out)):
	if out[i]['gender'] == None:
		prior[i] = 0.5
	elif out[i]['gender']=='male':
		prior[i] = out[i]['probability']
	elif out[i]['gender']=='female':
		prior[i] = 1-out[i]['probability']
height = df.crew_anthropometry_height_[:(df.crew_id_name_!='').sum()]
bayes_factor = pheightm(height)/((1-prior)*pheightf(height)+prior*pheightm(height))
posterior = prior*bayes_factor

with plt.style.context('fivethirtyeight'):
	plt.figure()
	plt.barh(np.arange(len(posterior)), (posterior-0.5)*2, color='cornflowerblue', lw=0,
			height=0.9)
	plt.xlabel(r"$\leftarrow$ Probability Female | Probability Male $\rightarrow$    ")
	for i in range(len(posterior)):
		plt.annotate(firstname[i], (0, np.arange(len(posterior))[i]+0.2), size='xx-small',
				ha='center', va='bottom')
	plt.yticks([])
	plt.xticks([-1, -0.5, 0, 0.5, 1],  ['1', '0.5', '0', '0.5', '1'])
	plt.subplots_adjust(bottom=0.1)
	plt.savefig("figs/names_bar")

#Male participant mask
male = posterior >0.5

#Correct where this is wrong
male[firstname=='Sage'] = True



#Participant anthropometry distribution

super = ['Height', 'Shoulder Width', 'Thumb-tip reach']
sub = ['All', 'Male', 'Female']
masks = [np.ones(len(df), bool), np.where(male), np.where(-male)]

f = open('anthropometry.txt', 'w')
for i in range(3):
	print super[i]
	print '\t   Min. 5th  50th 95th Max.'
	f.write(super[i]+'\n')
	f.write('\t   Min. 5th  50th 95th Max.\n')
	for k in range(3):
		subframe = df[df >0].ix[:, 19+i].ix[masks[k]]
		print '%10s %3.1f %3.1f %3.1f %3.1f %3.1f' % (sub[k], subframe.min(), subframe.quantile(0.05), subframe.quantile(0.5),  subframe.quantile(0.95),subframe.max())
		f.write('%10s %3.1f %3.1f %3.1f %3.1f %3.1f\n' % (sub[k], subframe.min(), subframe.quantile(0.05), subframe.quantile(0.5),  subframe.quantile(0.95),subframe.max()))
f.close()

with plt.style.context('fivethirtyeight'):
	plt.figure()
	h = np.linspace(62, 78)
	df[df >0].crew_anthropometry_height_.plot('hist', normed=True, label='Participcants', bins=np.arange(62, 78, 1))
	#plt.plot(h, 0.25/(np.sqrt(2*np.pi)*2.0)*np.exp(-(h-63.8)**2/2.0**2)+0.75/(np.sqrt(2*np.pi)*3.1)*np.exp(-(h-69.8)**2/3.1**2), label='75% male')
	plt.plot(h, 0.5/(np.sqrt(2*np.pi)*2.0)*np.exp(-(h-63.8)**2/2.0**2)+0.5/(np.sqrt(2*np.pi)*3.1)*np.exp(-(h-69.8)**2/3.1**2), label='US Population')
	plt.legend(loc=0)
	plt.xlabel('Height (in)')
	plt.ylabel('Probability density (in$^{-1}$)')
	plt.subplots_adjust(left=0.15, bottom=0.1)
	plt.savefig("figs/height_hist")

with plt.style.context('fivethirtyeight'):
	plt.figure()
	df[df >0].crew_anthropometry_thumbtip_.plot('hist', normed=True, label='Participcants')
	plt.xlabel('Thumb-tip reach (in)')
	plt.ylabel('Probability density (in$^{-1}$)')
	plt.subplots_adjust(left=0.15, bottom=0.1)
	plt.savefig("figs/thumbtip_hist")

with plt.style.context('fivethirtyeight'):
	plt.figure()
	df[df >0].crew_anthropometry_shoulderwidth_.plot('hist', normed=True, label='Participcants')
	plt.xlabel('Shoulder width (in)')
	plt.ylabel('Probability density (in$^{-1}$)')
	plt.subplots_adjust(left=0.15, bottom=0.1)
	plt.savefig("figs/shoulderwidth_hist")

with plt.style.context('fivethirtyeight'):
	plt.figure()
	df[df >0].crew_experience_age_.plot('hist', normed=True, label='Participcants')
	plt.xlabel('Participant age (years)')
	plt.ylabel('Probability density (year$^{-1}$)')
	plt.subplots_adjust(left=0.15, bottom=0.1)
	plt.savefig("figs/age_hist")

#Japanese female/American male
jf = np.array([[61.8, 15.3, 28.2],
				[1.9454,	0.7903,	1.5198]])
am = np.array([[70.8,	19.3,	32.1],
				[2.4318,	1.0335,	1.5806]])

ansur_f = np.array([58.5, 14.9, 25.6])
ansur_m = np.array([76.6, 22.1, 35.8])


h2 = np.arange(58, 80, 0.1)
sw = np.arange(13.5, 23, 0.1)
ttr = np.arange(24, 38, 0.1)

plt.figure(figsize=(8, 10))
for i in range(3):
	with plt.style.context('fivethirtyeight'):
		plt.subplot(3, 1, i+1)
		plt.xlabel(['Height', 'Shoulder width', 'Thumb-tip reach'][i]+" (in)")
		#plt.plot([h2, sw, ttr][i], 100*stats.norm.cdf([h2, sw, ttr][i], jf[0, i], jf[1, i]),
		#		label='Japanese Female', lw=3)
		#plt.plot([h2, sw, ttr][i], 100*stats.norm.cdf([h2, sw, ttr][i],  am[0, i], am[1, i]),
		#		label='American Male', lw=3)
		#plt.axvspan(df.ix[:, 19+i].min(), df.ix[:, 19+i].max(), color='lightgray', alpha=0.5)
		df.ix[:, 19+i].plot(kind='hist', normed=True, label='Participants',
					cumulative=False)
		if i==0:
			plt.plot(h2, 0.5/(np.sqrt(2*np.pi)*2.0)*np.exp(-(h2-63.8)**2/2.0**2)+0.5/(np.sqrt(2*np.pi)*3.1)*np.exp(-(h2-69.8)**2/3.1**2), label='US Population')
			plt.legend(loc=2, frameon=True, fontsize='small')

		plt.ylabel("")
		#if i==1:
		plt.ylabel("Probability density (in$^{-1})$", size='small')
		#plt.ylabel("Percentile")
		#plt.axis('tight')

		#if i==2:
		#	plt.xlabel("inches")
		orioncol = 'orange'
		isscol = 'coral'
		plt.axvline([58.6, 14.0, 25.7][i], linestyle='-', color=isscol, lw=2)
		plt.axvline([74.8, 20.9, 34.7][i], linestyle='-', color=isscol, lw=2)
		plt.axvline(ansur_f[i], linestyle='-', color=orioncol, lw=2)
		plt.axvline(ansur_m[i], linestyle='-', color=orioncol, lw=2)

		plt.annotate('5th pct. Japanese female',
				(stats.norm.ppf(0.05, jf[0, i], jf[1, i]), 0.12),
				va='top', ha='left', rotation=0, color='w',
				size='small', weight='bold', bbox=dict(boxstyle="round", lw=0, fc=isscol, alpha=0.75))
		plt.annotate('95th pct. American male',
				(stats.norm.ppf(0.95, am[0, i], am[1, i]), 0.17),
				 va='top', ha='right', rotation=0, color='w',
				  size='small', weight='bold', bbox=dict(boxstyle="round", lw=0, fc=isscol, alpha=0.75))
		plt.annotate('Orion Upper Limit', (ansur_m[i], 0.10),
				va='bottom', ha='right', rotation=0, color='w',
				size='small', weight='bold', bbox=dict(boxstyle="round", lw=0, fc=orioncol, alpha=0.75))
		plt.annotate('Orion Lower Limit', (ansur_f[i], 0.05),
				va='bottom', ha='left', rotation=0, color='w',
				size='small', weight='bold', bbox=dict(boxstyle="round", lw=0, fc=orioncol, alpha=0.75))
		if i ==1:
			plt.xlim(13, 23)
		plt.subplots_adjust(left=0.15, bottom=0.1, hspace=0.33)
	#plt.annotate("ANSUR: US Army ANthropometry SURvey", (0.9, 0.02), ha='right', va='baseline',
	#		xycoords="figure fraction", size='xx-small', color='darkgrey')
#plt.tight_layout()
plt.savefig("hist_percentiles")



#Privacy

colors = plt.cm.RdBu(np.linspace(0.25, 0.75, 6))

df[df > 0].ix[:, (dic['Type_of_Data']=="Ordinal")*(dic['Privacy']=='1')].hist(histtype='barstacked',
			bins=np.arange(1, 8), align='left', grid=False, normed=True, figsize=(12, 12))
plt.suptitle("Privacy")
plt.savefig("figs/ordinal_bar_privacy")

subframe = df[df > 0].ix[:, (dic['Type_of_Data']=="Ordinal")*(dic['Privacy']=='1')]
plt.figure()
for j in range(subframe.shape[1]):
	width = np.zeros(6)
	plt.subplot(subframe.shape[1], 1, j+1)
	left = 0
	total = np.in1d(subframe.ix[:, j], np.arange(1, 7)).sum()
	for k in range(6):
		width[k] = (subframe.ix[:, j]==k+1).sum()
		if (width[k]/total*100) > 5:
			plt.annotate('%2.0f%%' % (width[k]/total*100), ((left+width[k]/2)/total, 0.5),
					ha='center', va='center')
		plt.gca().add_patch( matplotlib.patches.Rectangle((left/total, 0),
					width[k]/total, 1, color=colors[k]))
		left +=width[k]
	plt.yticks([])
	plt.axis('tight')
	plt.axis('off')
	plt.title("\n".join(wrap(dic['Question'][(dic['Type_of_Data']=="Ordinal")*(dic['Privacy']=='1')][j], 88)), size='x-small')

	plt.subplots_adjust(hspace=1)
plt.suptitle('Privacy', size='large')
plt.axes([0.25, 0.03, 0.5, 0.05])
for j in range(1, 7):
	plt.gca().add_patch( matplotlib.patches.Rectangle((j, 0),
						0.5, 1, color=colors[j-1]))
	plt.annotate(j, (j+0.25, 0.25), ha='center', va='bottom')
	plt.axis('tight')
	plt.axis('off')
plt.savefig("figs/ordinal_stack_privacy")
plt.close()

#Stacked gauge charts

colors = plt.cm.RdBu(np.linspace(0.25, 0.75, 6))
for i in range(1, 14):
	mask = ((dic['Type_of_Data']=="Ordinal")+(dic['Type_of_Data']=="Categorical")+(dic['Type_of_Data']=="Binary"))*(dic['Location']==locs[i])
	subframe = df[df > 0].ix[:, mask]

	plt.figure(figsize=(8, 1.2*subframe.shape[1]+1))
	plt.subplot(subframe.shape[1]+1, 1, subframe.shape[1]+1)
	for j in range(1, 7):
		plt.gca().add_patch( matplotlib.patches.Rectangle((j, 0),
							0.5, 1, color=colors[-j]))
		plt.annotate(j, (7-j+0.25, 0.5), ha='center', va='center', color='#2a3439', size='large')
		plt.axis('tight')
		plt.axis('equal')
		plt.axis('off')
	for j in range(subframe.shape[1]):
		nlines = len(dic['Type_of_Data'][mask][j])/72.
		if dic['Type_of_Data'][mask][j]=="Ordinal":
			ticks = re.split('1 - | 6 - ', dic['Value'][mask][j])
			width = np.zeros(6)
			plt.subplot(subframe.shape[1]+1, 1, j+1)
			left = 0
			total = np.in1d(subframe.ix[:, j], np.arange(1, 7)).sum()
			for k in range(6):
				width[k] = (subframe.ix[:, j]==(6-k)).sum()
				if (width[k]/total*100) > 5:
					plt.annotate('%2.0f%%' % (width[k]/total*100), ((left+width[k]/2)/total, 0.5),
							ha='center', va='center')
				plt.gca().add_patch( matplotlib.patches.Rectangle((left/total, 0),
							width[k]/total, 1, color=colors[::-1][k]))
				left +=width[k]
			#plt.annotate(ticks[0], (0.5, 0), va='top', size='xx-small', ha='center')
			plt.annotate('1 - '+ticks[1], (1.0, 0), va='top', size='xx-small', ha='right', color='maroon')
			plt.annotate('6 - '+ticks[2], (0.0, 0), va='top', size='xx-small', ha='left', color='navy')
			plt.annotate('%2.1f' % subframe.ix[:, j].mean(), (0, 0.5), va='center', ha='right', size='large',
					bbox=dict(boxstyle="circle", lw=0, fc=plt.cm.RdBu((subframe.ix[:, j].mean()-1)/10.+0.25)))
			#plt.annotate('Mean', (0, 0.85), va='top', ha='right', size='xx-small', color='#2a3439')
			plt.yticks([])
			plt.axis('tight')
			plt.axis('off')
		#plt.suptitle(locs[i], size='large')
		#plt.axes([0.25, 0.03, 0.5, 0.05])
		elif dic['Type_of_Data'][mask][j]=="Binary":
			ticks = ['yes', 'no']
			width = np.zeros(2)
			plt.subplot(subframe.shape[1]+1, 1, j+1)
			left = 0
			total = np.in1d(subframe.ix[:, j], ['y', 'n']).sum()
			for k in range(2):
				width[k] = (subframe.ix[:, j]==['y', 'n'][k]).sum()
				if (width[k]/total*100) > 5:
					plt.annotate('%2.0f%%' % (width[k]/total*100), ((left+width[k]/2)/total, 0.5),
							ha='center', va='center')
				plt.gca().add_patch( matplotlib.patches.Rectangle((left/total, 0),
							width[k]/total, 1, color=colors[::5][::-1][k]))
				left +=width[k]
			#plt.annotate(ticks[0], (0.5, 0), va='top', size='xx-small', ha='center')
			plt.annotate(ticks[1], (1.0, 0), va='top', size='xx-small', ha='right', color='maroon')
			plt.annotate(ticks[0], (0.0, 0), va='top', size='xx-small', ha='left', color='navy')
			plt.yticks([])
			plt.axis('tight')
			plt.axis('off')
		elif dic['Type_of_Data'][mask][j]=="Categorical":
			ticks = re.split(' ', dic['Value'][mask][j])[1::2]
			width = np.zeros(2)
			plt.subplot(subframe.shape[1]+1, 1, j+1)
			left = 0
			total = np.in1d(subframe.ix[:, j], ticks).sum()
			total = subframe.ix[:, j].str.contains(ticks[0], case=False).sum()
			total += subframe.ix[:, j].str.contains(ticks[1], case=False).sum()
			for k in range(2):
				width[k] = subframe.ix[:, j].str.contains(ticks[k], case=False).sum()
				if (width[k]/total*100) > 5:
					plt.annotate('%2.0f%%' % (width[k]/total*100), ((left+width[k]/2)/total, 0.5),
							ha='center', va='center')
				plt.gca().add_patch( matplotlib.patches.Rectangle((left/total, 0),
							width[k]/total, 1, color=colors[::5][::-1][k]))
				left +=width[k]
			#plt.annotate(ticks[0], (0.5, 0), va='top', size='xx-small', ha='center')
			plt.annotate(ticks[1], (1.0, 0), va='top', size='xx-small', ha='right', color='maroon')
			plt.annotate(ticks[0], (0.0, 0), va='top', size='xx-small', ha='left', color='navy')
			plt.yticks([])
			plt.axis('tight')
			plt.axis('off')

		try:
			plt.title("\n".join(wrap(dic['Question'][mask][j], 80)), size='small', loc='left')
		except:
			plt.title(dic['Code'][mask][j], size='x-small')
		plt.subplots_adjust(hspace=1)
	plt.savefig("figs/ordinal_stack_"+locs[i])
	plt.close()


for i in range(1, 14):
	subframe = df[df > 0].ix[:, (dic['Type_of_Data']=="Ordinal")*(dic['Location']==locs[i])]
	subframe.hist(histtype='bar', bins=np.arange(1, 8), normed=True, align='left', grid=False, figsize=(12, 12))
	plt.suptitle('%s' % (locs[i]), size='large')
	plt.savefig("figs/ordinal_hist_"+locs[i])
	plt.close()

#Anthropometry, age, response correlations
corrstr = '\t\t\t\t\t\thgt    shl     rch    age    hrs\n'
print corrstr
f= open('correlations.txt', 'w')
for i in range(74):
    try:
		corr_row = '%40s' % dic['Code'][i]+ '\t'+ '%+0.2f  '*5 % (df.crew_anthropometry_height_.corr(df[df >0].ix[:, i], method='spearman'),
				df.crew_anthropometry_shoulderwidth_.corr(df[df >0].ix[:, i], method='spearman'),
				df.crew_anthropometry_thumbtip_.corr(df[df >0].ix[:, i], method='spearman'),
				df.crew_experience_age_.corr(df[df >0].ix[:, i], method='spearman'),
				df[df >0].crew_experience_aircraft_time.corr(df[df >0].ix[:, i], method='spearman'))
		print corr_row
		corrstr += corr_row+'\n'
    except:
        print
f.write(corrstr)
f.close()

'''for i in range(len(data[0])):
	try:
		plt.scatter(df[df >0].crew_anthropometry_height_, df[df >0].ix[:, i], lw=0)
		corr = df[df >0].crew_anthropometry_height_.corr(df[df >0].ix[:, i], 'spearman')
		plt.waitforbuttonpress()
		plt.clf()
		nvalid = (df[df >0].ix[:, i]==df[df >0].ix[:, i]).sum()
		z = 0.5*np.log((1+corr)/(1-corr))*np.sqrt((nvalid-3)/1.06)
		plt.title(dic['Code'][i]+' %0.2f'*2 % (corr, z))

	except:
		continue'''

#List responses
categories = ["All", "Male", "Female", "Flight Experience", ">40 Hours Flight",
				 "No Flight Experience", "21 or older", "Younger than 21",
				  "30 or older", "Younger than 30",
				 "Shorter than 70\"", "70\" or taller", "Aerospace Major",
				 "Above Limits", "Below Limits",
				 "CM1", "CM2", "CM3", "CM4"]
male2 = np.zeros(len(df), bool)
male2[np.where(male)[0]] = True

masks  = [np.ones(len(df), bool), male2, -male2*(np.ones(len(df), bool) < len(male)),
			 df[df >0].crew_experience_aircraft_=='y', df.crew_experience_aircraft_time > 40,
			df[df >0].crew_experience_aircraft_=='n',
			df[df >0].crew_experience_age_ >= 21,
			df[df >0].crew_experience_age_ < 21,
			df.crew_experience_age_ >= 30,
			df.crew_experience_age_ < 30,
			df[df >0].crew_anthropometry_height_ <70, df[df >0].crew_anthropometry_height_>=70,
			df.crew_experience_major_.str.contains('ASEN'),
			(df.crew_anthropometry_height_ > 74.8)+(df.crew_anthropometry_shoulderwidth_ > 20.9)+(df.crew_anthropometry_thumbtip_ > 34.7),
			(df.crew_anthropometry_height_ < 58.6)+(df.crew_anthropometry_shoulderwidth_ < 14.0)+(df.crew_anthropometry_thumbtip_ < 25.7),
			df[df >0].crew_id_number_=='1',
			df[df >0].crew_id_number_=='2',
			df[df >0].crew_id_number_=='3',
			df[df >0].crew_id_number_=='4']
mask = (dic['Type_of_Data']=="Ordinal")+(dic['Type_of_Data']=="Categorical")+(dic['Type_of_Data']=="Binary")+(dic['Type_of_Data']=="Count")
subframe = df[df > 0].ix[:, mask]
for i in range(len(subframe.T)):
	print dic['Location'][mask][i]
	print dic['Question'][mask][i]
	if dic['Type_of_Data'][mask][i] == 'Ordinal':
		print "Category\t\tn\tMean\t1\t2\t3\t4\t5\t6\t(4-6)\tp-value"
		for j in range(len(categories)):
			width = np.zeros(6)
			total = np.in1d(subframe.ix[:, i].ix[masks[j]], np.arange(1, 7)).sum()
			for k in range(6):
				width[k] = (subframe.ix[:, i].ix[masks[j]]==(k+1)).sum()*1./total
			pval = stats.ranksums(subframe.ix[:, i].ix[masks[j]].dropna(), subframe.ix[:, i].ix[-masks[j]].dropna())[1]
			print '%21s\t%i' % (categories[j], total) + '\t%2.1f' % (subframe.ix[:, i].ix[masks[j]]).mean() + '\t%3.1f%%'*6 % tuple(width*100)+'\t%3.1f%%' % (width[3:].sum()*100) +'\t%3.2f' % (pval)+'*'*(pval < 0.05)
		print
	elif dic['Type_of_Data'][mask][i] == 'Binary':
		print "Category\t\t n\tyes\t no\tp-value"
		for j in range(len(categories)):
			yes = subframe.ix[:, i].ix[masks[j]].str.contains('y').sum()
			no = subframe.ix[:, i].ix[masks[j]].str.contains('n').sum()
			total = yes+no
			if categories[j] !='All':
				p0 = subframe.ix[:, i].ix[-masks[j]].str.contains('y').sum()*1./(subframe.ix[:, i].ix[-masks[j]].str.contains('y').sum()+subframe.ix[:, i].ix[-masks[j]].str.contains('n').sum())
				pval = stats.binom_test((yes, no), p=p0)
			else:
				pval = np.nan
			print '%21s\t%i' % (categories[j], total) + '\t%3.1f%%'*2 % (yes*1./total*100, no*1./total*100)+'\t%3.2f' % (pval)+'*'*(pval < 0.05)
		print
	elif (dic['Type_of_Data'][mask][i] == 'Categorical') *(i>2):

		responsetypes = dic['Value'][mask][i].split()[1::2]
		print "Category\t\t n\t"+ responsetypes[0]+"\t"+responsetypes[1]+"\t no\tp-value"
		for j in range(len(categories)):
			suc = subframe.ix[:, i].ix[masks[j]].str.contains(responsetypes[0], case=False).sum()
			fail = subframe.ix[:, i].ix[masks[j]].str.contains(responsetypes[1], case=False).sum()
			total = suc+fail
			if categories[j] !='All':
				p0 = subframe.ix[:, i].ix[-masks[j]].str.contains(responsetypes[0], case=False).sum()*1./(subframe.ix[:, i].ix[-masks[j]].str.contains(responsetypes[1], case=False).sum()+subframe.ix[:, i].ix[-masks[j]].str.contains(responsetypes[0], case=False).sum())
				pval = stats.binom_test((suc, fail), p=p0)
			else:
				pval = np.nan
			print '%21s\t%i' % (categories[j], total) + '\t%3.1f%%'*2 % (suc*1./total*100, fail*1./total*100)+'\t%3.2f' % (pval)+'*'*(pval < 0.05)
		print
	elif (dic['Type_of_Data'][mask][i] == 'Count')*(i > 5):

		responsetypes = np.unique(subframe.ix[:, i].dropna())
		print "Category\t\tn\t"+"Mean"+ "\t%s"*len(responsetypes) % tuple(responsetypes)+ "\tp-value"
		for j in range(len(categories)):
			width = np.zeros(len(responsetypes))
			total = np.in1d(subframe.ix[:, i].ix[masks[j]], responsetypes).sum()
			for k in range(len(responsetypes)):
				width[k] = (subframe.ix[:, i].ix[masks[j]]==responsetypes[k]).sum()*1./total
			pval = stats.ranksums(subframe.ix[:, i].ix[masks[j]].dropna(), subframe.ix[:, i].ix[-masks[j]].dropna())[1]
			print '%21s\t%i' % (categories[j], total) + '\t%2.1f' % (subframe.ix[:, i].ix[masks[j]]).mean() + '\t%3.1f%%'*len(responsetypes) % tuple(width*100)+'\t%3.2f' % (pval)+'*'*(pval < 0.05)
		print




#Ordinal response by location

polarity = np.zeros(13)
rating_volume = np.zeros(13)
rating_layout = np.zeros(13)
response_rate = np.zeros(13)
rating = np.zeros(13)

plt.figure(figsize=(16, 9))
for i in range(1, 14):
    plt.subplot(4, 4, i)
    df[df > 0].ix[:, (dic['Type_of_Data']=="Ordinal")*(dic['Location']==locs[i])].mean(0).plot(kind='barh', lw=0)
    polarity[i-1] = df[df > 0].ix[:, (dic['Type_of_Data']=="Ordinal")*(dic['Location']==locs[i])].mean(0)[df[df > 0].ix[:, (dic['Type_of_Data']=="Ordinal")*(dic['Location']==locs[i])].mean(0) > 4].mean()
    rating_volume[i-1] = (df[df > 0].ix[:, (dic['Type_of_Data']=="Ordinal")*(dic['Location']==locs[i])*(dic['Reverse']=='0')*(dic['Volume']=='1')] <6).mean().mean()
    rating_layout[i-1] = (df[df > 0].ix[:, (dic['Type_of_Data']=="Ordinal")*(dic['Location']==locs[i])*(dic['Reverse']=='0')*(dic['Layout']=='1')] <6).mean().mean()
    rating[i-1] = df[df > 0].ix[:, (dic['Type_of_Data']=="Ordinal")*(dic['Location']==locs[i])*(dic['Reverse']=='0')].mean().mean()
    response_rate[i-1] =((rf !='-999')*(rf !='-888')*(rf !='')).ix[:, dic['Location']==locs[i]].mean().mean()

    plt.title(locs[i])
    plt.yticks(size='xx-small', rotation=-45, ha='right', va='bottom')
plt.subplots_adjust(left=0.1, right=0.98, wspace=0.6, hspace=0.3)
plt.savefig("figs/ordinal_bar")


#Polarity spatial mapping

def spatial(data, mask, cmap=plt.cm.Oranges, vmin=None, vmax=None, title='', label='', fmt='\n%2.0f%%', **args):
	plt.figure(figsize=(7.5, 10))
	rects = np.array([[240, -85.5+240, 88, 171],
						[0, -60.5+240, 240, 121],
						[240+44, -85.5, 88/2., 171],
						[240, 0, 88/2., 171/2.],
						[240, -171/2., 88/2., 171/2.],
						[240*3/4., 0, 240/4., 60.5],
						[240*3/4., -60.5, 240/4., 60.5],
						[240*1/2., 0, 240/4., 60.5],
						[240*1/2., -60.5, 240/4., 60.5],
						[0, -60.5, 240/2., 121]])
	abbr = ['nde', 'hab', 'eva', 'hyg', 'exr', 'cmd', 'gly', 'co2', 'sci', 'slp']
	names = ['Node', 'Habitation Module', 'EVA', 'Hygiene', 'Exercise', 'Command',
				'Galley', 'ECLSS', 'Science', 'Sleep']
	if vmin ==None:
		vmin = data.min()
	if vmax==None:
		vmax = data.max()
	rating = np.zeros(len(abbr))
	for i in range(len(rects)):
		total = (mask*(dic['Location']==abbr[i])).sum()
		#rating[i] = (7-data.ix[:, mask*(dic['Location']==abbr[i])*(dic['Reverse']=='1')]).mean().sum()/total
		'''if (dic['Type_of_Data']=="Ordinal").all():
			npos = (mask*(dic['Location']==abbr[i])*(dic['Reverse']=='0')).sum()
			nneg = (mask*(dic['Location']==abbr[i])*(dic['Reverse']=='1')).sum()
			pos = data.ix[:, mask*(dic['Location']==abbr[i])*(dic['Reverse']=='0')].mean().mean()*npos
			neg = (7-data.ix[:, mask*(dic['Location']==abbr[i])*(dic['Reverse']=='0')].mean().mean())*nneg
			rating[i] = (npos*pos+neg*nneg)/(npos+nneg)
			print neg
		else:'''
		rating[i] = data.ix[:, mask*(dic['Location']==abbr[i])].mean().mean()
		#rating[i] = data.ix[:, mask*(dic['Location']==abbr[i])*(dic['Reverse']=='0')].mean().mean()
		shade = (rating[i]-vmin)/(vmax-vmin)
		plt.gca().add_patch( matplotlib.patches.Rectangle((rects[i][0], rects[i][1]),
							rects[i][2], rects[i][3], color=cmap(shade)))
		if np.sum(cmap(shade)) < 2.5:
			textcolor='0.9'
		else:
			textcolor = '0.1'
		if rating[i]==rating[i]:
			plt.annotate(names[i]+fmt % rating[i], (rects[i][0]+rects[i][2]/2, rects[i][1]+rects[i][3]/2),
							ha='center', va='center',  fontweight='bold', size='small', color=textcolor)
		else:
			plt.annotate(names[i]+'\nn/a', (rects[i][0]+rects[i][2]/2, rects[i][1]+rects[i][3]/2),
							ha='center', va='center',  fontweight='bold', size='small', color=textcolor)
	plt.title(title)
	plt.axis('equal')
	plt.axis('off')
	cbar= plt.axes([0.25, 0.475, 0.5, 0.05], frameon=False)
	cbar.imshow([cmap(np.linspace(0, 1))], extent=(vmin, vmax, 0, (vmin-vmax)), aspect=0.1)
	cbar.set_yticks(ticks=[])
	cbar.set_xlabel(label)
	return abbr, rating

mask = (dic['Type_of_Data']=="Ordinal")*(dic['Volume']=='1')*(dic['Preference']=='0')
input = 100-100*((df[df >0] > 3)*(dic['Reverse']=='1')+(df[df >0] < 4)*(dic['Reverse']=='0'))
spatial(input, mask, vmin=70, vmax=100, label="%", title='Ratings 4 or higher  (volume)',
		cmap=plt.cm.Blues)
plt.savefig("figs/ordinal_map_volume")

mask = (dic['Type_of_Data']=="Ordinal")*(dic['Layout']=='1')*(dic['Preference']=='0')
spatial(input, mask, vmin=70, vmax=100, cmap=plt.cm.Blues, label="%", title='Ratings 4 or higher (layout)')
plt.savefig("figs/ordinal_map_layout")

mask = (dic['Type_of_Data']=="Ordinal")*(dic['Preference']=='0')
spatial(input, mask, vmin=70, vmax=100, cmap=plt.cm.Blues, label="%", title='Ratings 4 or higher')
plt.savefig("figs/ordinal_map_both")

mask = (dic['Type_of_Data']=="Ordinal")*(dic['Preference']=='0')
df2 = DataFrame(df[df >0], copy=True)
for i in range(67):
	for j in np.where(dic['Reverse']=='1')[0]:
		df2.set_value(i, j, 7-df[df >0].ix[i, j], True)
spatial(df2[df >0], mask, vmin=4.8, vmax=5.7, cmap=plt.cm.Blues, label="",
	title='Mean Ordinal Response', fmt='\n%2.1f')
plt.savefig("figs/ordinal_map_overall")

mask = dic['Freeresponse']=='1'
spatial(((rf!='-888')*(rf!='-999')*(rf!=''))*100, mask, vmin=20, vmax=80, cmap=plt.cm.coolwarm, label="%",
	title='Comment Rate')
plt.savefig("figs/response_rate_map")


with plt.style.context('ggplot'):
	plt.figure()
	plt.hist(egress['Time_in_seconds'][egress_mask], 6, color='cornflowerblue', normed=True)
	plt.ylabel('Density (s$^{-1}$)')
	plt.xlabel('Egress time (s)')
	plt.savefig("figs/egress_hist")

plt.close('all')
overall = DataFrame()
longlocs = ['', 'Entire Mock-up', 'Command Station', 'ECLSS', "Storage",  'Emergency Egress',
	'EVA', 'Exercise', 'Galley', 'Habitation Module', 'Hygiene', 'Node', 'Science', 'Sleep']
print "Loc. t   v   l"
headers = ['Total', 'Volume', 'Layout']
for i in range(1, len(locs)):
    print locs[i], '%2.1f '*3 % (df2.ix[:, (locs[i]==dic['Location'])*(dic['Type_of_Data']=="Ordinal")*(dic['Preference']=='0')].mean().mean(),
    	df2.ix[:, (locs[i]==dic['Location'])*(dic['Type_of_Data']=="Ordinal")*(dic['Preference']=='0')*(dic['Volume']=='1')].mean().mean(),
    	df2.ix[:, (locs[i]==dic['Location'])*(dic['Type_of_Data']=="Ordinal")*(dic['Preference']=='0')*(dic['Layout']=='1')].mean().mean())
for i in range(1, len(locs)):
	for j in range(3):
		mask2 = [np.ones(len(dic['Volume']), bool), (dic['Volume']=='1'), (dic['Layout']=='1')]
		overall.set_value(headers[j], longlocs[i],
			df2.ix[:, (locs[i]==dic['Location'])*(dic['Type_of_Data']=="Ordinal")*(dic['Preference']=='0')*mask2[j]].mean().mean())
with plt.style.context('fivethirtyeight'):
	overall.T.sort_index(by='Total').Total.plot(kind='barh', legend=False, xlim=(1, 6))
	plt.tight_layout()
	plt.axvline(overall.T.Total.median(), color='k')
	plt.annotate('Median\n %2.1f' % overall.T.Total.median(), (overall.T.Total.median(), 6),
				va='center', ha='left')
	plt.savefig("figs/overall_bar")

changed = np.genfromtxt('F.R. Location Classification - Sheet2.tsv', names=True,
		dtype=(np.object, np.int, np.int, np.int))
changes = DataFrame(changed, index=np.array(longlocs[1:])[np.argsort(np.array(locs)[1:][np.argsort(changed['Location'])])])

with plt.style.context('fivethirtyeight'):
	plt.figure(figsize=(18, 6))

	plt.subplot(131)
	(changes.overall_ich_volume_excess/(rf.overall_ich_volume_excess!='-888').sum()*100).plot(kind='barh')
	plt.title("Is there any place where there \nwas more volume than needed?")
	plt.xlabel("% of responses")
	plt.subplot(132)
	(changes.overall_ich_volume_deficit/(rf.overall_ich_volume_deficit!='-888').sum()*100).plot(kind='barh')
	plt.title("Is there any place where\n more volume is needed?")
	plt.xlabel("% of responses")
	plt.subplot(133)
	(changes.overall_ich_layout_/(rf.overall_ich_layout_!='-888').sum()*100).plot(kind='barh')
	plt.title("Would you change the layout\n or arrangement of locations?")
	plt.tight_layout()
	plt.xlabel("% of responses")
	plt.savefig("figs/changes")
