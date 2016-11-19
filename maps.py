from load_CHAMP import *

def spatial(data, mask, cmap=plt.cm.Oranges, vmin=None, vmax=None, title='', label='', fmt='\n%2.0f%%', **args):
	plt.figure(figsize=(7.5, 10))
	rects = np.array([[240, 85.5, 63, 0],
						[0, -60.5+240, 240, 121],
						[240+44, -85.5, 88/2., 171],
						[240, 0, 88/2., 171/2.],
						[240, -171/2., 88/2., 171/2.],
						[240*3/4., 0, 240/4., 60.5],
						[240*3/4., -60.5, 240/4., 60.5],
						[240*1/2., 0, 240/4., 60.5],
						[240*1/2., -60.5, 240/4., 60.5],
						[0, -60.5, 240/2., 121]])
	abbr = ['alk', 'cmd', 'co2', 'emr', 'exr', 'gly', 'hab', 'hch', 'hyg', 'sci', 'slp', 
			'str', 'tec', 'wdw']
	names = ['Airlock', 'Command', 'ECLSS', 'Emergency Path', 'Exercise', 'Galley', 
				'Habitat', 'Hatches', 'Science', 'Sleep', 'Storage', 
				'Technology Development', 'Windows']
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
