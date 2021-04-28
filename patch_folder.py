import numpy as np
import glob as glob

path = "/home/Drive3/yashashwi/tcga_REMAINING/thyroid_remaining/patches/"
l1=(np.linspace(0,26,27))
l2=(np.linspace(100,109,10))
l=np.hstack((l1,l2))
l.astype(int)

files_list=[]
for i in range(0,50):
	files_list.append([])


# print(len(files_list))
for i in range(0,len(l)):
	path_f=path+str(int(l[i]))+'/'
	files = (glob.glob(path_f+"*/*.png"))
	# print(len(files))
	# print(files[0])
	print(i)
	if (len(files)>0):
		print(files[0])
	for j in range(0,len(files)):
		a=(files[j].split('_'))
		index=(a[6].split('.'))[0]
		# print(index)
		files_list[int(index)].append(files[j])
	

print(len(files_list[0]))