import os, cv2, dlib, keras, glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from keras.models import Sequential,Model
from keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K

path = '.'
dnnFaceDetector=dlib.get_frontal_face_detector()

def crop_train():
	image_path_names=[]
	global person_names
	person_names=set()
	for file_name in os.listdir('./Images/'):
	    image_path_names.append('./Images/' + file_name)
	    person_names.add(file_name.split('_')[0])

	if os.path.exists(path+'/Images_crop2/') == False:
		os.mkdir(path+'/Images_crop2/')
	# os.mkdir(path+'/Images_crop2/')
	for person in person_names:
		if os.path.exists(path+'/Images_crop2/'+person+'/') == False:
  			os.mkdir(path+'/Images_crop2/'+person+'/')	
	

	for file_name in image_path_names:
		img=cv2.imread(file_name)
		gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		print('processing for', file_name)
		rect=dnnFaceDetector(gray,1)
		left,top,right,bottom=0,0,0,0
		
		if(rect):
			for (i,rect) in enumerate(rect):
				left=rect.left() #x1
				top=rect.top() #y1
				right=rect.right() #x2
				bottom=rect.bottom() #y2

			width=right-left
			height=bottom-top

			img_crop=img[top:top+height,left:left+width]
			img_path=path+'/Images_crop2/'+file_name.split('/')[-1].split('_')[0]+'/'+file_name.split('/')[-1]
			cv2.imwrite(img_path,img_crop)
		
		else:
			continue	


def crop_test():
	test_image_path_names=[]
	for file_name in os.listdir(path+'/Images_test/'):
	  test_image_path_names.append(file_name)
  
	if os.path.exists(path+'/Test_Images_crop/') == False:
		os.mkdir(path+'/Test_Images_crop/')


	for person in person_names:
		if os.path.exists(path+'/Test_Images_crop/'+person+'/') == False:
			os.mkdir(path+'/Test_Images_crop/'+person+'/')

	for file_name in test_image_path_names:
		img=cv2.imread('./Images_test/'+ file_name)
		print('procesing for', file_name)
		gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rects=dnnFaceDetector(gray,1)
		left,top,right,bottom=0,0,0,0

		if(rects):
			for (i,rects) in enumerate(rects):
				left=rects.left() #x1
				top=rects.top() #y1
				right=rects.right() #x2
				bottom=rects.bottom() #y2

			width=right-left
			height=bottom-top
			img_crop=img[top:top+height,left:left+width]
			img_path=path+'/Test_Images_crop/'+file_name.split('/')[-1][:-7]+'/'+file_name.split('/')[-1]
			cv2.imwrite(img_path,img_crop)
			
		else:
			continue



def make_model():
	global model
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))


	model.summary()
	model.load_weights('./vgg_face_weights.h5')

# def augmentation():
# 	global x

# 	datagen = ImageDataGenerator(
#         rotation_range=40,
#         zoom_range=0.2,
#         horizontal_flip = True)

# 	for person in tqdm(os.listdir('Images_crop2/')):
# 		os.mkdir('augmentated/train/'+person)
# 		for imgs in os.listdir('Images_crop2/'+person):
# 			x = load_img('Images_crop2/'+ person +'/'+imgs)
# 			x = img_to_array(x)
# 			x = x.reshape((1,) + x.shape)

# 			i = 0
# 			for batch in datagen.flow(x, batch_size=1,
# 			    save_to_dir=('augmentated/train/'+person), save_prefix=imgs, save_format='jpeg'):
# 			    i += 1
# 			    if i > 20:
# 			        break   

# 	for person in tqdm(os.listdir('Test_Images_crop/')):
# 		os.mkdir('augmentated/test/'+person)
# 		for imgs in os.listdir('Test_Images_crop/'+person):
# 			x = load_img('Test_Images_crop/'+ person +'/'+imgs)
# 			x = img_to_array(x)
# 			x = x.reshape((1,) + x.shape)

# 			i = 0
# 			for batch in datagen.flow(x, batch_size=1,
# 			    save_to_dir=('augmentated/test/'+person), save_prefix=imgs, save_format='jpeg'):
# 			    i += 1
# 			    if i > 20:
# 			        break 

from tqdm import tqdm
# Prepare Training Data
def prepare_training_data():
	global x_train, y_train, person_rep
	x_train=[]
	y_train=[]

	person_folders=os.listdir(path+'/Images_crop2/')
	person_rep=dict()
	for i,person in tqdm(enumerate(person_folders)):
		person_rep[i]=person
		image_names=os.listdir('./Images_crop2/'+person+'/')
		# print(person, image_names)	
		for image_name in image_names:
			print('processing for', image_name)
			img=load_img(path+'/Images_crop2/'+person+'/'+image_name,target_size=(224,224))
			img=img_to_array(img)
			img=np.expand_dims(img,axis=0)
			img=preprocess_input(img)
			img_encode=vgg_face.predict(img)
			x_train.append(np.squeeze((img_encode)).tolist())
			y_train.append(i)




def prepare_testing_data():
	global x_test,y_test

	x_test=[]
	y_test=[]
	person_folders=os.listdir(path+'/Test_Images_crop/')
	for i,person in tqdm(enumerate(person_folders)):
		image_names=os.listdir(path+'/Test_Images_crop/'+person+'/')
		for image_name in image_names:
			img=load_img(path+'/Test_Images_crop/'+person+'/'+image_name,target_size=(224,224))
			img=img_to_array(img)
			img=np.expand_dims(img,axis=0)
			img=preprocess_input(img)
			img_encode=vgg_face.predict(img)
			x_test.append(np.squeeze((img_encode)).tolist())
			y_test.append(i)

######################
#calling functions

# crop_train()
# crop_test()

# augmentation()

make_model()
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

prepare_training_data()
prepare_testing_data()

# print(len(x_train))
# print(person_rep)

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

print(x_train.shape)
print(x_test.shape)

# np.save('x_train', x_train)
# np.save('x_test', x_test)
# np.save('y_train', y_train)
# np.save('y_test', y_test)


# np.load('./numpy'+x_train, x_train)
# np.load('./numpy'+x_test, x_test)
# np.load('./numpy'+y_train, y_train)
# np.load('./numpy'+y_test, y_test)

############################


def model2():
	global classifier_model

	classifier_model=Sequential()
	classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
	classifier_model.add(BatchNormalization())
	classifier_model.add(Activation('tanh'))
	classifier_model.add(Dropout(0.3))
	classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
	classifier_model.add(BatchNormalization())
	classifier_model.add(Activation('tanh'))
	classifier_model.add(Dropout(0.2))
	classifier_model.add(Dense(units=3,kernel_initializer='he_uniform'))
	classifier_model.add(Activation('softmax'))
	classifier_model.compile(loss='sparse_categorical_crossentropy' ,optimizer='nadam',metrics=['accuracy'])			


model2()
classifier_model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))

classifier_model.save('face_major2.h5')
vgg_face.save('vgg_face.h5')

# predictions()

# keras.models.save_model(classifier_model, 'face_major2.h5')

############################(
def predictions():
	if os.path.exists(path+'/Predictions/') == False:
		os.mkdir(path+'/Predictions/')

	for img_name in os.listdir('./testing/'):
	  
	  if (img_name != '.ipynb_checkpoints'):
	    print('processing for', img_name)
	    img=cv2.imread('./testing/' +img_name)
	    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    # Detect Faces
	    rects=dnnFaceDetector(gray,1)
	    
	    if(rects):
		    left,top,right,bottom=0,0,0,0
		    for (i,rects) in enumerate(rects):
		      # Extract Each Face
		      left=rects.left() #x1
		      top=rects.top() #y1
		      right=rects.right() #x2
		      bottom=rects.bottom() #y2
		      width=right-left
		      height=bottom-top
		      img_crop=img[top:top+height,left:left+width]
		      cv2.imwrite(path+'/Images_test/crop_img.jpg',img_crop)
		      
		      # Get Embeddings
		      crop_img=load_img(path+'/Images_test/crop_img.jpg',target_size=(224,224))
		      crop_img=img_to_array(crop_img)
		      crop_img=np.expand_dims(crop_img,axis=0)
		      crop_img=preprocess_input(crop_img)
		      embed=vgg_face.predict(crop_img)

		      # Make Predictions
		      # embed=K.eval(img_encode)
		      person=classifier_model.predict(embed)
		      print(type(person))
		      print(person)
		      name=person_rep[np.argmax(person)]
		      os.remove(path+'/Images_test/crop_img.jpg')
		      temp = (str(np.max(person)))
		      print(temp)
		      # print('start1')
		      # print(temp.split('.')[1][:3])

		      # if(temp.split('.')[1][:3] > '83'):
		      	

		      cv2.rectangle(img,(left,top),(right,bottom),(255,0,0), 2)
		      img=cv2.putText(img,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2,cv2.LINE_AA)
		      img=cv2.putText(img,str(np.max(person)),(right,bottom+10),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2,cv2.LINE_AA)

		      cv2.imwrite(path+'/Predictions/'+img_name,img)


predictions()




