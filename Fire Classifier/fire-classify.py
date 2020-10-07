import cv2
import os
import sys
import math
import numpy as np

import tflite_runtime.interpreter as tflite

# tflite - load model

#print("Load tflite model from: firenet.tflite ...", end = '')
tflife_model = tflite.Interpreter(model_path="inceptionv1onfire.tflite")
tflife_model.allocate_tensors()
print("OK")

# Get input and output tensors.
tflife_input_details = tflife_model.get_input_details()
tflife_output_details = tflife_model.get_output_details()

################################################################################

# load video file

video = cv2.VideoCapture('cropfire.mp4')

# get video properties

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_counter = 0
fail_counter = 0

while (True):

    # get video frame from file, handle end of file

    ret, frame = video.read()
    if not ret:
        print("... end of video file reached")
        break

   # print("frame: " + str(frame_counter),  end = '')
    frame_counter = frame_counter + 1

    # re-size image to network input size and perform prediction

    # input to networks is: 224x224x3 colour image with channel ordering as {B,G,R}
    # as is the opencv norm, not {R,G,B} and pixel value range 0->255 for each channel

    small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)

    ############################################################################

    np.set_printoptions(precision=6)

    # perform prediction with tflite model via TensorFlow

    tflife_input_data = np.reshape(np.float32(small_frame), (1, 224, 224, 3))
    tflife_model.set_tensor(tflife_input_details[0]['index'], tflife_input_data)

    tflife_model.invoke()

    output_tflite = tflife_model.get_tensor(tflife_output_details[0]['index'])
   # print("\t: TFLite (via tensorflow): ", end = '')
 #  print(output_tflite)
    

    if round(output_tflite[0][0]) == 1:
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
        cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
        cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);

    else:
        cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
        cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
        cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);    
    
    start_t = cv2.getTickCount();

    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

    windowName = "Live Fire Detection - FireNet CNN"
    cv2.imshow(windowName, frame);

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time = round(1000/fps);


    key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
    if (key == ord('x')):
        keepProcessing = False;
    elif (key == ord('f')):
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    elif (key == ord('q')):
        break
else:
    print("usage: python firenet.py videofile.ext");

video.release()
cv2.destroyAllWindows()  
