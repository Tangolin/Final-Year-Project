"""
The GAIT-IT dataset provides various gait representations useful for gait pathology analysis: 
i) sequences of binary silhouettes; 
ii) sequences of skeletons; 
iii) GEIs; 
iv) SEIs. 
A GEI and SEI are available for each gait cycle, as well as for the complete set of gait cycles available per sequence.
The spatial dimension of the produced gait representations is 224*224. However, the computation of gait representations
is done with the full resolution of the captured gait sequences, to preserve information. Cropping removes the
background around the subject's bounding box and then the width of the cropped image is padded to match its height,
while maintaining the centroid position. Finally, the square image is resized to 224*224 pixels, while maintaining the
aspect ratio. All representations consider a 10 fps framerate.
"""

import math
import os

import cv2
import numpy as np

root_dir = "./Core Data/INIT_GaitDB/silhouettes"


def find_centroid(image):
    """Simple function to find the centroid of a contour shape."""
    contours, _ = cv2.findContours(image, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_LIST)
    cnt = max(contours, key=cv2.contourArea)

    # Obtain the moment of the contours
    moments = cv2.moments(cnt)
    cx, cy = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])

    return cx, cy


for folder in os.listdir(root_dir):
    sub_path = os.path.join(root_dir, folder)
    output_imgs = []

    for file in os.listdir(sub_path):
        fp = os.path.join(sub_path, file)
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)

        # Find the contours, image is already binarized
        contours, hierarchy = cv2.findContours(
            img, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_LIST
        )
        cnt = max(
            contours, key=cv2.contourArea
        )  # Find the contour with the greatest area

        # Find bounding rectangle and resize image
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_img = img[y : y + h, x : x + w]
        height, width = cropped_img.shape
        scale_ratio = 224 / height

        # Avoid an unbalanced cropping problem later
        adj_width, adj_height = (
            math.floor(scale_ratio * width),
            round(scale_ratio * height),  # Handle decimal issue
        )
        if (224 - adj_width) % 2 != 0:
            adj_width += 1

        resized_img = cv2.resize(
            cropped_img,
            (adj_width, adj_height),
        )

        # Pad image to create a 224 * 224 image
        padding = (224 - resized_img.shape[1]) / 2
        padded_img = cv2.copyMakeBorder(
            resized_img,
            0,
            0,
            int(padding),
            int(padding),
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        output_imgs.append(padded_img)

        # img_with_contours = cv2.imread(fp, cv2.IMREAD_COLOR)

        # # Draw contours on the copied image
        # cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

        # # Draw a bounding box on the copied image
        # cv2.rectangle(img_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # Draw a dot on the copied image for centre of contour
        # cv2.circle(
        #     img_with_contours, (cx, cy), radius=0, color=(0, 0, 255), thickness=-1
        # )
        # # Display the image with contours
        # cv2.imshow("Image with Contours", resized_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    ref_cx, ref_cy = find_centroid(output_imgs[0])
    final_img = output_imgs[0]
    img_count = len(output_imgs)
    alpha = 1 / img_count

    for count, img in enumerate(output_imgs[1:]):
        # Repeat the same steps to obtain the contour in the new image
        cx, cy = find_centroid(img)

        # Find the adjustment factor and do transformation
        dx, dy = ref_cx - cx, ref_cy - cy
        rows, cols = img.shape
        transform_mat = np.float32([[1, 0, dx], [0, 1, 0]])
        t_img = cv2.warpAffine(img, transform_mat, (cols, rows))

        # Add the image together
        if count == 0:
            final_img = cv2.addWeighted(final_img, alpha, t_img, alpha, 0)
        else:
            final_img = cv2.addWeighted(final_img, 1, t_img, alpha + 0.001, 0)

    save_path = os.path.join("./Core Data/INIT_GaitDB/GEI", folder + "_GEI.png")
    cv2.imwrite(save_path, final_img)
