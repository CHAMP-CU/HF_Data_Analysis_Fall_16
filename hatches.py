from load_CHAMP import *

hatches = DataFrame(index = ['First', 'Second', 'Third'], 
		columns=[u'PCM forward (to HAB)', u'HAB aft (to PCM)', u'HAB forward (to airlock)'])

operated = -DataFrame(columns=hatches.columns, index=df.index, dtype=bool)
for i in np.where(-df.review_hch_filter.isnull())[0]:
        operated.ix[i][df.review_hch_filter[i].split(', ')] =True

hatches.ix[0] = df.review_hch_rank_01[operated.ix[:, 0]].value_counts()[hatches.columns]
hatches.ix[1] = df.review_hch_rank_02[operated.ix[:, 1]].value_counts()[hatches.columns]
hatches.ix[2] = df.review_hch_rank_03[operated.ix[:, 2]].value_counts()[hatches.columns]

hatches[::-1].plot(kind='barh', stacked=True)
plt.xlabel("Number of Responses")
plt.savefig('../results/figs/hatches')