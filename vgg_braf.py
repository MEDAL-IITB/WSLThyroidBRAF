# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
import keras.backend as K
import cv2
import glob
import copy
import pandas
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

df=pd.read_csv('braf_tcga_patients_label_used_final.csv')
braf_s=list(df.Actual_Label)
filenames=list(df.Patient)


model_vgg16_conv = VGG16(include_top=False, weights='imagenet')
model_vgg16_conv.summary()


input = Input(shape=(2500,2500,3))


output_vgg16_conv = model_vgg16_conv(input)

my_model = Model(input=input, output=output_vgg16_conv)

my_model.summary()

datagen_test = ImageDataGenerator(
    rescale=1.0/255)

print(filenames)

dire_pos="pos/"
dire_neg="neg/"

for i in range(len(filenames)):

	try:
		dire="/home/Drive3/yashashwi/tcga_REMAINING/thyroid_remaining/patches/"+str(i)+'/'
		train_generator = datagen_train.flow_from_directory(
	    dire,
	    batch_size=1,
	    target_size =(2500,2500),
	    class_mode='categorical')

		f=train_generator.filenames

		if(len(f)>0):

			fname=f[0].split("__")[2]

			print(fname)

			exit()
			for k in range(0,len(filenames)):
				if(fname==filenames[k]):
					label_patient=braf_s[k]
			# fname_final=fname_split[0]+"_"+fname_split[2]+"_"+"label"+str(label_patient)
			fname_final=fname+"_"+"label"+str(label_patient)
			# print(label_patient)
			# if(label_patient!=-1)
			feats=my_model.predict_generator(train_generator)
			# print("Saving bags")
			bag_f=np.reshape(feats,[len(feats)*78*78,1,512])
			# if(len(feats)>20):
			# 	pp=0
			# 	aa=True
			# 	while(aa):
			# 		try:
			# 	# for pp in range(0,len(feats),20):
			# 			# n_instances=round(len(feats)*78*78/20)
			# 			bag_pp=bag_f[pp:pp+20]
			# 			if(label_patient==0):
			# 				np.save(dire_neg+str(pp)+fname_final+".npy",bag_pp)
			# 			if(label_patient==1):
			# 				np.save(dire_pos+str(pp)+fname_final+".npy",bag_pp)
			# 			pp=pp+20
			# 			print(pp)
			# 		except:
			# 			aa=False
			# 			bag_pp=bag_f[pp+20:len(feats)]
			# 			if(label_patient==0):
			# 				np.save(dire_neg+str(pp)+fname_final+".npy",bag_pp)
			# 			if(label_patient==1):
			# 				np.save(dire_pos+str(pp)+fname_final+".npy",bag_pp)
			# else:
			if(label_patient==0):
				np.save(dire_neg+fname_final+".npy",bag_f)
			if(label_patient==1):
				np.save(dire_pos+fname_final+".npy",bag_f)

			print('Saved')
	except:
		continue
		# print(feats.shape)
	

	# np.save("feats_vgg_orig.npy",feats)
	# np.save("fnames_vgg_orig.npy",f)

# print(f)


#Then training with your data ! 
