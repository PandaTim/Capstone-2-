import os
import sys
import argparse
import dlib
import cv2
from tqdm import tqdm
from icrawler.builtin import GoogleImageCrawler
from imutils.face_utils.helpers import shape_to_np
from imutils.face_utils import rect_to_bb

"""
Example: python crawl_face.py -k [KEYWORD] --num [NUMBER OF IMAGE] -o [OUTPUT_DIRECTORY]
"""
parser = argparse.ArgumentParser(description='crawl + facial landmark detection')
parser.add_argument('-k', '--keyword', dest='keyword', default='') # your keyword goes here ...
parser.add_argument('--num', dest='crawl_num', default=250) # the number of images you want to crawl goes here ...
parser.add_argument('-o', '--output', dest='crawl_output', default='result')
args = parser.parse_args()

key_word = args.keyword
crawl_num = args.crawl_num
crawl_output = args.crawl_output

if not os.path.exists(crawl_output):
    os.mkdir(crawl_output)
if not os.path.exists(crawl_output + "/" + key_word):
    os.mkdir(crawl_output + "/" + key_word)
if not os.path.exists(crawl_output + "/" + key_word + "/face"):
    os.mkdir(crawl_output + "/" + key_word + "/face")
google_crawler = GoogleImageCrawler(storage={'root_dir': crawl_output + "/" + key_word})
google_crawler.crawl(keyword=key_word, max_num=crawl_num)

print("Crawling complete ... Initialize facial detection")
img_list = os.listdir(crawl_output + "/" + key_word)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('predictor.dat')
count = 0
for i in tqdm(range(len(img_list)), file=sys.stdout, position=0):
    if img_list[i][-3:] != 'png' and img_list[i][-3:] != 'jpg':
        continue
    img = cv2.imread(crawl_output + "/" + key_word + "/" + img_list[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 0)
    count += len(rects)
    tqdm.write(img_list[i] + ": " + str(len(rects)) + "faces found")
    for (index, rect) in enumerate(rects):
        (bX, bY, bW, bH) = rect_to_bb(rect)
        cv2.rectangle(img, (bX, bY), (bX + bW, bY + bH),
                      (0, 255, 0), 2)
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(img_gray, rect)
        shape = shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
        for (j, (x, y)) in enumerate(shape):
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.imwrite(crawl_output + "/" + key_word + "/face/" + img_list[i][:-4] + "_face_" + str(index) + ".jpg",
                    img[bY:bY + bH, bX:bX + bW])
    cv2.imwrite(crawl_output + "/" + key_word + "/" + img_list[i], img)
print(count, "faces are found in ", crawl_num, "images")
