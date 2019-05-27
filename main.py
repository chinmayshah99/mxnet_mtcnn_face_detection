# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

detector = MtcnnDetector(model_folder='model', ctx=mx.gpu(), num_worker = 4 , accurate_landmark = True)



def main_logic(file_name):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    print(file_name)
    temp_file_name = file_name.rsplit('.', 1)[0]
    print(temp_file_name)

    opdirname = 'out/'

    camera = cv2.VideoCapture('inp/' + file_name)
    print(camera.isOpened()) # checking if file is opened

    if camera.isOpened():

    
     
        # With webcam get(CV_CAP_PROP_FPS) does not work.
        # Let's see for ourselves.
        
        if int(major_ver)  < 3 :
            fps = int(camera.get(cv2.cv.CV_CAP_PROP_FPS))
        else :
            fps = int(camera.get(cv2.CAP_PROP_FPS))

        no_frames = int(camera.get(5)) # CV_CAP_PROP_FRAME_COUNT

        next_frame = 0
        for i in range(no_frames):
            camera.set(1, next_frame) #CV_CAP_PROP_POS_FRAMES
            next_frame += fps # get frame every 1s
            
            grab, frame = camera.read()
            img = cv2.resize(frame, (320,180)).copy()

            t1 = time.time()
            results = detector.detect_face(img)

            if results is not None:

                total_boxes = results[0]
                points = results[1]
                
                # extract aligned face chips
                chips = detector.extract_image_chips(img, points, 144, 0.37)
                for i, chip in enumerate(chips):
                    # cv2.imshow('chip_'+str(i), chip)
                    print(os.path.join(opdirname,'chip_'+ temp_file_name+str(i)+str(t1) +'.png'))
                    cv2.imwrite(os.path.join(opdirname,'chip_'+ temp_file_name+str(i)+str(t1) +'.png'), chip)

                draw = img.copy()
                for b in total_boxes:
                    cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

                for p in points:
                    for i in range(5):
                        cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
                print(os.path.join(opdirname,'chip_draw'+temp_file_name+str(i)+str(t1) +'.png'))
                cv2.imwrite(os.path.join(opdirname,'chip_draw'+temp_file_name+str(i)+str(t1) +'.png'), draw)

                if no_frames == next_frame:
                    break

    else:
        assert "couldn't open file"



for root, dirs, files in os.walk("inp"):  
    for filename in files:
        main_logic(filename)
