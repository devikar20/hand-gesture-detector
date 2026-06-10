import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


from notifypy import Notify 


#open the webcam
import cv2 as cv
import numpy
cap = cv.VideoCapture(0)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


a=False
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global a
    if result.hand_landmarks:
     finger=(result.hand_landmarks[0][8])
     knuckle=(result.hand_landmarks[0][5])
    
     if finger.y<knuckle.y:
         print("open hand")
         if a==False:
           notification = Notify()
           notification.title = "open hand"
           notification.message = "This is a custom desktop notification."


           notification.send()
           a=True
     else:
           if a==True:
            print("fist")
            a=False
            
           
    

options = HandLandmarkerOptions(
     base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
     running_mode=VisionRunningMode.LIVE_STREAM,
     result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker:
    
    
    while True:

       ret, frame = cap.read()
       ms = time.time_ns() // 1_000_000
      
       
    
       
       if ret==True:
          gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
          cv.imshow('frame', frame)
          mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
          landmarker.detect_async(mp_image, ms)
          
          
          
    
          if cv.waitKey(1) & 0xFF == ord('q'):
            break
       else:
          print("oh no")
          break
    
   
  

cap.release()
cv.destroyAllWindows()