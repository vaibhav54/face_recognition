import cv2, time, dlib
import numpy as np

from keras.preprocessing.image import load_img,img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

classifier_model = load_model('./face_major2.h5')
vgg_face = load_model('./vgg_face.h5')
person_rep = dict()
person_rep = {0: 'Pranjal', 1: 'Prince', 2: 'Vaibhav'}
print(person_rep)
video = cv2.VideoCapture(0)
dnnFaceDetector=dlib.get_frontal_face_detector()
# global crop

while True:
	check, frame = video.read()

	print('check')
	# print(frame)
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects=dnnFaceDetector(gray,1)
	# cv2.imshow('original',gray)
	cv2.imwrite('./video_frame/crop_img.jpg',gray)

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
	      img_crop=frame[top:top+height,left:left+width] 

	      cv2.imwrite('./video_frame/final.jpg',img_crop)
	      crop_img=load_img('./video_frame/final.jpg',target_size=(224,224))
	      crop_img=img_to_array(crop_img)
	      crop_img=np.expand_dims(crop_img,axis=0)
	      crop_img=preprocess_input(crop_img)
	      embed=vgg_face.predict(crop_img)

	      print(type(crop_img))
	      print(img_crop.shape)
	 
	      person=classifier_model.predict(embed)
	      temp = (str(np.max(person)))
	      print(person)
	      if(float(temp) < 0.90):
	      	name = '*unknown face*'
	      else:	
	      	name=person_rep[np.argmax(person)]
		  # name=person_rep[np.max(person)]
	      # os.remove('./final.jpg')
	      
	      print(temp)


	      cv2.rectangle(gray,(left,top),(right,bottom),(255,0,0), 2)
	      img=cv2.putText(gray,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
	      img=cv2.putText(img,str(np.max(person)),(right,bottom+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
	      
	      cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0), 2)
	      img2=cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
	      img2=cv2.putText(img2,str(np.max(person)),(right,bottom+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
	      
	      # cv2.imwrite('./video_frame/cropped_img.jpg',img_crop)
	      cv2.imshow('crop',gray)
	      cv2.imshow('frame',frame)
	      print('end')
	      # crop = img_crop
	# cv2.imshow('img',crop)


	# cv2.imshow('img2',gray)
	key = cv2.waitKey(1)
	if key == ord('q'):
		break

video.release()
cv2.destroyAllWindows()