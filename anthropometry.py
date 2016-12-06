from load_CHAMP import *

ansur_men = pd.read_csv('ansur_men.txt', delimiter='\t')
ansur_women = pd.read_csv('ansur_women.txt', delimiter='\t')
pars = ['STATURE', 'BIDELTOID_BRTH', 'THUMB-TIP_REACH']

orioncol = 'orange'
isscol = 'coral'
jf = np.array([[61.8, 15.3, 28.2],
				[1.9454,	0.7903,	1.5198]])
am = np.array([[70.8,	19.3,	32.1],
				[2.4318,	1.0335,	1.5806]])

ansur = [ansur_men, ansur_women]
plt.figure(figsize=(15, 15))
for i in range(3):
    for j in range(2):
		plt.subplot(3, 2, 2*(i+1)+j-1)
		anth = [df.crew_height, df.crew_shoulder, df.crew_thumb][i][subcat_gender.ix[:, j+1]]
		anth.hist(normed=True, label='Test Data')
		(ansur[j][pars[i]]/25.4).hist(normed=True, histtype='step', lw=3, label='ANSUR Ref. Data')
		plt.title(['Male', 'Female'][j]+' '+['Stature', 'Bideltoid Breadth', 'Thumb-tip Reach'][i])
		stat, pval=   stats.ttest_ind(anth, ansur[j][pars[i]]/25.4)
		bias = anth.mean()-(ansur[j][pars[i]]/25.4).mean()
		plt.annotate('Bias: %3.1f $\pm$ %3.1f in \np-value: %3.2f' % (bias, bias/stat, pval), (1, 1),
					xycoords='axes fraction', va='top', ha='right', bbox=dict(boxstyle="round", lw=0, fc='white', alpha=0.75))

		plt.xlabel("in")
		plt.ylabel("Density (in$^{-1}$)")
		if 2*(i+1)+j-1==1:
			plt.legend(loc=2, fontsize='small')
		plt.subplots_adjust(hspace=0.5, wspace=0.3, left=0.1)
        #mark orion and iss limits
		plt.axvline([58.6, 14.0, 25.7][i], linestyle='-', color=isscol, lw=2)
		plt.axvline([74.8, 20.9, 34.7][i], linestyle='-', color=isscol, lw=2)
		plt.axvline([58.5,14.9,25.6][i], linestyle='-', color=orioncol, lw=2)
		plt.axvline([76.6,22.1,35.8][i], linestyle='-', color=orioncol, lw=2)
		plt.annotate('5th pct. \nJpn. female',
					(stats.norm.ppf(0.05, jf[0, i], jf[1, i]), 0.85*1/(np.sqrt(2*np.pi)*jf[1,i]**2)),
					va='top', ha='left', rotation=0, color='w',
					size='small', weight='bold', bbox=dict(boxstyle="round", lw=0, fc=isscol, alpha=0.75))
		plt.annotate('95th pct. \nAm. male',
					(stats.norm.ppf(0.95, am[0, i], am[1, i]),  0.95*1/(np.sqrt(2*np.pi)*am[1,i])),
					va='top', ha='right', rotation=0, color='w',
					size='small', weight='bold', bbox=dict(boxstyle="round", lw=0, fc=isscol, alpha=0.75))
		plt.annotate('Orion Upper Limit', ([76.6,22.1,35.8][i], 0.6*1/(np.sqrt(2*np.pi)*am[1,i])),
					va='bottom', ha='right', rotation=0, color='w',
					size='small', weight='bold', bbox=dict(boxstyle="round", lw=0, fc=orioncol, alpha=0.75))
		plt.annotate('Orion Lower Limit', ([58.5,14.9,25.6][i],  0.6*1/(np.sqrt(2*np.pi)*jf[1,i])),
					va='bottom', ha='left', rotation=0, color='w',
					size='small', weight='bold', bbox=dict(boxstyle="round", lw=0, fc=orioncol, alpha=0.75))
    if i ==1:
        	plt.xlim(13, 23)
    plt.subplots_adjust(left=0.15, bottom=0.1, hspace=0.33)

plt.savefig("../results/figs/anthropometry_bias")

plt.close()


#Anthropometry, age, experience, response correlations
corrstr = '\t\t\t\t\t\tn\thgt    shl     rch    age\n'
print corrstr
f= open('../results/tables/correlations.txt', 'w')
for i in np.where(dic['Data_type']=='Ordinal')[0]:
		corr_row = '%40s' % dic['Fall_2016_Question_Code'][i]+ '\t%2i'% df.ix[:, i].count()+ '\t'+ '%+0.2f  '*4 % (df.crew_height.corr(df.ix[:, i], method='spearman'),
				df.crew_shoulder.corr(df.ix[:, i], method='spearman'),
				df.crew_thumb.corr(df.ix[:, i], method='spearman'),
				df.crew_age.corr(df.ix[:, i], method='spearman'))
		print corr_row
		corrstr += corr_row+'\n'
f.write(corrstr)
f.close()
