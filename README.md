# face_recognition
Face detection & recognition. I used 17 images(14 train & 3 test) for each class. I made use of dlib library for detecting faces, and image encoding with the help of pre-trained network VGG_face_net which outputs 2622 embedding of each cropped face.<br> <br>
Result:<br>
<img src ="Predictions/test3.png" width=30%>

#### Face_recognition in webcam for real-time recognition:
System recognizes face if the model predicts it with greater than 90% accuracy, otherwise label of 'unknown_face' is displayed.<br>
Result:<br>
<img src ="result_from_webcam/Screenshot%20(361).png" width=70%>

<br> <br>

Download the vgg_face weights:
https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo
## Reference:
https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5
