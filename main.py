# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import argparse


'''
    The function exists to ensure that in case of side photo, the box takes into account negative values
    this function makes sure that in case of negative values are encountered, it makes them zero
    
    this point can be sometimes negative---> ----
                                            |   |
                                            -----
'''
def negative_zero(value):
    if value < 0:
        return 0
    else:
        return value


def main_logic(detector, img_name, input_dir, opdirname):
    print(img_name)
    img = cv2.imread(input_dir + '/'+ img_name)
    temp_img_name = img_name.rsplit('.', 1)[0]
    # print(temp_img_name)
    # run detector
    results = detector.detect_face(img)
    opdirname = opdirname + '/'
    # print(results)
    if results is not None:

        total_boxes = results[0]
        points = results[1]
        
        # # extract aligned face chips from image and write image for each cropped face
        # chips = detector.extract_image_chips(img, points, 144, 0.37)
        # for i, chip in enumerate(chips):
        #     # cv2.imshow('chip_'+str(i), chip)
        #     print(os.path.join(opdirname,'chip_'+ temp_img_name+str(i)+'.png'))
        #     cv2.imwrite(os.path.join(opdirname,'chip_'+ temp_img_name+str(i)+'.png'), chip)

        draw = img.copy()
        counter = 0
        for b in total_boxes:
            # print(b)
            # b = [x,y, x1,y1, confidence%]
            if b[4] > 0.97: # making sure face with 97% + accuracy are only detected
                cropped_image = draw[  negative_zero(int(b[1])) : negative_zero(int(b[3])), negative_zero(int(b[0])): negative_zero(int(b[2]))]
                cv2.imwrite(os.path.join(opdirname,'chip_'+ temp_img_name+str(counter)+'.png'), cropped_image)
                counter+=1
                # to draw rectangle in the main image
                # cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

        # to draw the 5 points on face
        # for p in points:
        #     for i in range(5):
        #         cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

        # print(os.path.join(opdirname,'chip_draw'+temp_img_name+str(counter)+'.png'))
        # cv2.imwrite(os.path.join(opdirname,'chip_draw'+temp_img_name+str(counter)+'.png'), draw)
        # cv2.imshow("detection result", draw)
        # cv2.waitKey(0)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='inp', help='input directory for image')
    parser.add_argument('-o', '--output_path', type=str, default='out', help='output directory for image')
    parser.add_argument("-g", "--gpu", action="store_true", help="True if to use GPU default CPU")

    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    input_dir = args.input_path
    output_dir = args.output_path

    if args.gpu:
        processor = mx.gpu()
    else:
        processor = mx.cpu()

    detector = MtcnnDetector(model_folder = 'model', ctx = processor, num_worker = 4 , accurate_landmark = True) 

    for root, dirs, files in os.walk(input_dir):        
        for file2 in files:
            main_logic(detector, file2, input_dir, output_dir)
        for dir in dirs:
            print(dir)
            for root1, dirs1, files1 in os.walk(input_dir + '/' + dir):
                print(files1)
                if not os.path.exists(output_dir + '/' + dir):
                    os.makedirs(output_dir + '/' + dir)

                for file2 in files1:
                    main_logic(detector, file2, input_dir + '/' + dir, output_dir + '/' + dir)