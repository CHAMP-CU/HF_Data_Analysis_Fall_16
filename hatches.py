from load_CHAMP import *

hatches = DataFrame(index = ['First', 'Second', 'Third'], 
		columns=[u'PCM forward (to HAB)', u'HAB aft (to PCM)', 
					u'HAB forward (to airlock)'])

operated = -DataFrame(columns=hatches.columns, index=df.index, dtype=bool)
for i in np.where(-df.review_hch_filter.isnull())[0]:
        operated.ix[i][df.review_hch_filter[i].split(', ')] =True
        
operated.sum(1).hist(bins=np.arange(5), align='left', label='Test Data')
pmf = stats.binom.pmf(np.arange(0, 5), 3, 0.56)*72
plt.plot(np.arange(0, 5), pmf, drawstyle='steps-mid', 
		label='Binomial Distribution\n($\pi$=0.56)')
plt.xlabel("Number of hatches operated")
plt.xlabel("Number of participants")
plt.xticks([0, 1, 2, 3])
plt.legend(loc=0, framealpha=0)
plt.savefig('../results/figs/hatches_operated')

hatches.ix[0] = df.review_hch_rank_01[operated.ix[:, 0]].value_counts()[hatches.columns]
hatches.ix[1] = df.review_hch_rank_02[operated.ix[:, 1]].value_counts()[hatches.columns]
hatches.ix[2] = df.review_hch_rank_03[operated.ix[:, 2]].value_counts()[hatches.columns]

allthree = DataFrame(index = ['First', 'Second', 'Third'], 
		columns=[u'PCM forward (to HAB)', u'HAB aft (to PCM)', 
					u'HAB forward (to airlock)'])
allthree.ix[0] = pd.get_dummies(df.review_hch_rank_01[(operated.sum(1) == 3)]).sum()[hatches.columns]
allthree.ix[1] = pd.get_dummies(df.review_hch_rank_02[(operated.sum(1) == 3)]).sum()[hatches.columns]
allthree.ix[2] = pd.get_dummies(df.review_hch_rank_03[(operated.sum(1) == 3)]).sum()[hatches.columns]


(allthree[::-1]/(operated.sum(1) == 3).sum()*100).plot(kind='barh', stacked=True, 
				width=1, legend=False, figsize=(12, 6))
plt.legend(ncol=3, framealpha=0.5)
plt.axis('tight')
plt.xlabel("%")
plt.savefig('../results/figs/hatch_contingiency')



hatches[::-1].plot(kind='barh', stacked=True)
plt.xlabel("Number of Responses")
plt.savefig('../results/figs/hatches')

combos = pd.get_dummies(df.review_hch_filter).astype(bool)

for i in range(7):
	counts = df.review_hch_rank_01.ix[np.where(combos.ix[:, i])].value_counts()[combos.ix[:, i].name.split(', ')]
	counts2 = df.review_hch_rank_02.ix[np.where(combos.ix[:, i])].value_counts()[combos.ix[:, i].name.split(', ')]
	if len(counts) == 1:
		continue
	elif len(counts)==2:
		print counts.argmax(), counts.max(), counts.argmin(), counts.min(), stats.binom_test(counts)
		print  stats.chi2_contingency(np.array([[counts], [counts2]]))[1]

