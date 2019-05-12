import my_yolov3_api as yolo
import torch
import numpy as np
import time
import cv2
import util

srcFolder:str = 'imgs'
destFolder:str = 'det'
yolo.load_model()
imgNames = yolo.file_names_from_folder(srcFolder)
# for i, imgnam in enumerate(imgNames):
#     print(i, imgnam )
imgNames = imgNames[:5]
images = yolo.load_images(imgNames)

# img = images[0]
# print(img.shape)
# img, magic = yolo.resize_image(img, (yolo.imageHeight, yolo.imageHeight))
# print(img.shape, magic)

imgs = images
imgs_ = imgs.copy()
imgs, factors = yolo.resize_images(imgs)
print(imgs.shape)
print(factors)
print()

imgs = yolo.prepare_tensor_image(imgs)
print(imgs.shape)
preds = yolo.predict_images(imgs)
boxes = yolo.post_process_predictions(preds)
boxes = yolo.select_objects(boxes, yolo.vehicles)
print(boxes)

for i in range(len(boxes)):
    plotted = yolo.draw_bbox(imgs_[i], boxes[i], factors[i])
    cv2.imwrite('det/temp/temp{}.jpg'.format(i),plotted)

# img_ = images[0].astype(np.uint8).copy()
# for clas, box in boxes.items():
#     print(clas, box)
#     box[:, [0,2]] -= magic[1]
#     box[:, [1,3]] -=magic[2]
#     box /= magic[0]
#     print(box)
#     # print(images[1].shape)
#     box = box.numpy().astype(int)
#     for b in box:
#             c1, c2 = tuple(b[:2]), tuple(b[2:4])
#             cv2.rectangle(img_, c1, c2, color=(200,0,200))
#             # cv2.putText(img_, classes[clas],\
#             #     (c1[0], c1[1] + 12 + 4),cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
# cv2.imwrite('temp.jpg',img_)

# images = yolo.preprocess_image(images)
# img_ = images.copy()
# # print(images.shape)
# # img_ = images[:,:,::-1].transpose((0,2,3,1)).copy()
# images = yolo.prepare_tensor_image(images)

# for i in range(len(images)):
#     print(imgNames[i:i+1])
#     img = images[i:i+1]
#     print(img.shape)
#     start = time.time()
#     prediction = yolo.predict_images(img)
#     print(time.time() - start)
#     print(prediction.shape)

#     print('///////////////////////////////////////////')
#     boxes = yolo.post_process_predictions(prediction)
#     yolo.resize_bbox(boxes[0], (img))
    # temp = (img_[i]).transpose((1,2,0))[:,:,::-1].copy()
    # # # ploted = yolo.draw_bbox(img[i], boxes[i])
    # ploted = yolo.draw_bbox2(temp, boxes[i])
    # # # print(temp.shape)
    # # ploted = temp.astype(np.uint8)
    # # # cv2.rectangle(ploted, (100,150), (150,250), color=(200,0,200), thickness=2)
    # cv2.imwrite('temp.jpg',ploted)
    
    # exit()
