import os

folder = '/home/ljzhang/data/sfada/office-home/'
domains = os.listdir(folder)
domains.sort()

for d in range(len(domains)):
	dom = domains[d]
	if os.path.isdir(os.path.join(folder, dom)):
		classes = os.listdir(os.path.join(folder, dom))
		classes.sort()
		print(classes)
		
		f = open(folder + dom+'/' + dom+"_list.txt", "w")
		for c in range(len(classes)):
			cla = classes[c]
			files = os.listdir(os.path.join(folder, dom, cla))
			files.sort()
			for file in files:
				print('{:} {:}'.format(os.path.join(cla, file), c))
				f.write('{:} {:}\n'.format(os.path.join(cla, file), c))
		f.close()