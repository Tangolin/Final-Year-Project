import os

import cv2

root_dir = "./Core Data/INIT_GaitDB/silhouettes"

for folder in os.listdir(root_dir):
    print(folder)
    sub_path = os.path.join(root_dir, folder)
    output_imgs = []

    for file in os.listdir(sub_path):
        fp = os.path.join(sub_path, file)
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("test", img)
        cv2.waitKey(0)
