# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

detector = MtcnnDetector(model_folder='model', ctx=mx.gpu(), num_worker = 4 , accurate_landmark = True)



def main_logic(img_name):
    print(img_name)
    img = cv2.imread(img_name)
    temp_img_name = img_name.rsplit('.', 1)
    print(temp_img_name)
    # run detector
    results = detector.detect_face(img)
    opdirname = 'out/'

    if results is not None:

        total_boxes = results[0]
        points = results[1]
        
        # extract aligned face chips
        chips = detector.extract_image_chips(img, points, 144, 0.37)
        for i, chip in enumerate(chips):
            # cv2.imshow('chip_'+str(i), chip)
            print(os.path.join(opdirname,'chip_'+ temp_img_name+str(i)+'.png'))
            cv2.imwrite(os.path.join(opdirname,'chip_'+ temp_img_name+str(i)+'.png'), chip)

        draw = img.copy()
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
        print(os.path.join(opdirname,'chip_draw'+temp_img_name+str(i)+'.png'))
        cv2.imwrite(os.path.join(opdirname,'chip_draw'+temp_img_name+str(i)+'.png'), draw)
        # cv2.imshow("detection result", draw)
        cv2.waitKey(0)

    # --------------
    # test on camera
    # --------------
    '''
    camera = cv2.VideoCapture(0)
    while True:
        grab, frame = camera.read()
        img = cv2.resize(frame, (320,180))

        t1 = time.time()
        results = detector.detect_face(img)
        print 'time: ',time.time() - t1

        if results is None:
            continue

        total_boxes = results[0]
        points = results[1]

        draw = img.copy()
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
        cv2.imshow("detection result", draw)
        cv2.waitKey(30)
    '''
for root, dirs, files in os.walk("inp"):  
    for filename in files:
        main_logic(filename)
