from keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
face_detection_videocam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# load our serialized face detector model from disk
label = os.path.sep.join([settings.BASE_DIR, "face_detector/labels.pickle"])
model = load_model(os.path.join(settings.BASE_DIR,'face_detector/face_keras_model3.h5'))
import pickle
with open(os.path.sep.join([settings.BASE_DIR, "face_detector/labels.pickle"]), 'rb') as f:
    labels = pickle.load(f)
    labels ={v:k for k,v in labels.items()}


class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, frame = self.video.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		if faces is():
			cv2.putText(frame, "no face found", (200,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
			frame_flip = cv2.flip(frame,1)
			ret, jpeg = cv2.imencode('.jpg', frame_flip)
		else:
			for(x,y,w,h) in faces:
				roi = gray[y:y+h, x:x+w]
				cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,240),2)
				crop_face = cv2.resize(roi, (200, 200))
				#print(crop_face.shape)
				data_p = np.asarray(crop_face, dtype=np.int32)
				data_p = data_p.reshape((data_p.shape[0], data_p.shape[1], 1))
				data_p = np.expand_dims(data_p, -1)
				#print(data_p.shape)
				output = np.argmax(model.predict(np.expand_dims(data_p, axis=0)).round(2))
				#print(output)
				name =labels[output]
				#print(name)
				cv2.putText(frame, name, (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
				frame_flip = cv2.flip(frame,1)
				ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()
