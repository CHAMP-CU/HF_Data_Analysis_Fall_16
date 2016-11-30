from load_CHAMP import *

ansur_men = pd.read_csv('ansur_men.txt', delimiter='\t')
ansur_women = pd.read_csv('ansur_women.txt', delimiter='\t')
pars = ['STATURE', 'BIDELTOID_BRTH', 'THUMB-TIP_REACH']

ansur = [ansur_men, ansur_women]
plt.figure(figsize=(10, 10))
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
					xycoords='axes fraction', va='top', ha='right')

		plt.xlabel("in")
		plt.ylabel("Density (in$^{-1}$)")
		if 2*(i+1)+j-1==1:
			plt.legend(loc=2, fontsize='small')
		plt.subplots_adjust(hspace=0.5, wspace=0.3, left=0.1)

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
