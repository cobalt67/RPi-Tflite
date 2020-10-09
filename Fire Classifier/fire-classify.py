import cv2
import os
import sys
import math
import numpy as np

import tflite_runtime.interpreter as tflite

# Load the Tflite model

model = tflite.Interpreter(model_path="fire-classifier.tflite")
model.allocate_tensors()

# Get input and output tensors.
tflife_input_details = model.get_input_details()
tflife_output_details = model.get_output_details()


# Load the Video 

video = cv2.VideoCapture(0)

# For storing video file on the device
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.mp4',fourcc, 3, (640,480))

# Get the Video properties

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_counter = 0
fail_counter = 0

while (True):

    # get video frame from file, handle end of file

    ret, frame = video.read()
  # ret = video.set(3,400)
  # ret = video.set(4,400)


    if not ret:
        print("error")
        break

    frame_counter = frame_counter + 1

    # re-size image to network input size and perform prediction

    # input to networks is: 224x224x3 colour image with channel ordering as {B,G,R}
    # as is the opencv norm, not {R,G,B} and pixel value range 0->255 for each channel

    small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)

    np.set_printoptions(precision=6)

    # perform prediction with tflite model via TensorFlow

    tflife_input_data = np.reshape(np.float32(small_frame), (1, 224, 224, 3))
    model.set_tensor(tflife_input_details[0]['index'], tflife_input_data)

    model.invoke()

    output_tflite = model.get_tensor(tflife_output_details[0]['index'])

    if round(output_tflite[0][0]) == 1:
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 3)
        cv2.putText(frame,'FIRE',(int(width/3),int(height/19)),
        cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA);

    else:
        cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 3)
        cv2.putText(frame,'CLEAR',(int(width/3),int(height/19)),
        cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA);    
    
    windowName = "Fire Detection for Edge Applications"
    cv2.imshow(windowName, frame);
    out.write(frame)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time = round(1/fps);

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()  
