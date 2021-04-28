import cv2
import openslide
import numpy as np
import os
from PIL import Image
import glob
#cls = "/luminalA/"
path = "/home/Drive3/yashashwi/tcga_REMAINING/color_norm/tcga_color_norm/new_norm"
dest = "/home/Drive3/yashashwi/tcga_REMAINING/thyroid_remaining/patches"
level = 1
# label = 2
slide = 1
window = 500*2*5
threshould = 240

def generate_patches(filename, patient,ind):
	# if not os.path.exists(dest+'/'+str(i)+'/'+patient+'/'):
	# 	# print("aaaaaaaaa")
	#     # os.mkdir(dest+'/'+patient+'/')
	# 	os.mkdir(dest+'/'+str(i)+'/'+patient+'/')
	ind=100+ind
	if not os.path.exists(dest+'/'+str(ind)+'/'):
		os.mkdir(dest+'/'+str(ind)+'/')
	if not os.path.exists(dest+'/'+str(ind)+'/'+patient+'/'):
		os.mkdir(dest+'/'+str(ind)+'/'+patient+'/')


	a = openslide.OpenSlide(filename)
	col,row = a.level_dimensions[level]
	# print(len(range(0,row-window,np.int(window/slide))))
	# print(len(range(0,col-window,np.int(window/slide))))
	kk=0
	for i in range(0,row-window,np.int(window/slide)):
		for j in range(0,col-window,np.int(window/slide)):
			im_object = a.read_region((j,i),level,(window,window))
			tpatch = np.asarray(im_object)
			patch = tpatch[:,:,:3]
			prob0 = patch[:,:,0].mean()
			prob1 = patch[:,:,1].mean()
			prob2 = patch[:,:,2].mean()
			if(prob0 < threshould and prob1 < threshould and prob2 < threshould):
				kk=kk+1
					# im_object.save(dest+'/'+patient+'/'+ patient + '__'+ str(i) +'_'+ str(j) + '.png')
				im_objectnew=im_object.resize((2500,2500))
				im_objectnew.save(dest+'/'+str(ind)+"/"+patient +'/'+patient +'__'+ str(kk) + '.png')
				# for qq in range(1,50):
				# 		angle=np.random.randint(1,359)
				# 		# rI=rotateImage(im_objectnew, angle)
				# 		im2 = im_objectnew.convert('RGBA')
				# 		# rotated image
				# 		rot = im2.rotate(angle, expand=0)
				# 		# a white image same size as rotated image
				# 		fff = Image.new('RGBA', rot.size, (255,)*4)
				# 		# create a composite image using the alpha layer of rot as a mask
				# 		out = Image.composite(rot, fff, rot)
				# 		# save your work (converting back to mode='1' or whatever..)
				# 		print(dest+'/'+str(ind)+'/'+patient+'/' +patient+'__'+ str(i) +'_'+ str(j) +'_'+str(qq)+ '.png')
				# 		out.convert(im_objectnew.mode).save(dest+'/'+str(ind)+'/'+patient+'/' +patient+'__'+ str(i) +'_'+ str(j) +'_'+str(qq)+ '.png')

import glob, os
# os.chdir("/home/SSD/thyroid_remaining")
# for file in glob.glob("*.txt"):
	# print(file)
# files = sorted(glob.glob(os.path.join(path, "*/*.svs")))
files = (glob.glob(path+"*/*.svs"))

# x, patient = files[0].split("TCGA-")
# print(patient)
print len(files)
# i=0
list_p=['TCGA-BJ-A45E','TCGA-BJ-A2N8','TCGA-BJ-A192']
for ind in range(0,len(files)):
	# i=i+1
	print(ind)
	x, patient = files[ind].split("TCGA-")
	patient, y = patient.split("-01Z")
	# print(patient)
	# print(file)
	if not os.path.exists((os.path.join(path,str("TCGA-"+patient)))):
		a = openslide.OpenSlide(files[ind])
		mag = a.properties["openslide.objective-power"]
		print(str("TCGA-"+patient))
		# generate_patches(files[ind], str(mag+"__"+"TCGA-"+patient),ind)

