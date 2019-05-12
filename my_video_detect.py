import my_yolov3_api as yolo
import cv2
import numpy as np

videofile = '/media/tsuman/98D2644AD2642EA6/vehicle_videos/DSC_0036.MOV'
cap = cv2.VideoCapture(videofile)
frames = -1
yolo.load_model()

while cap.isOpened():
    ret, frame = cap.read()
    frames +=1
    if frames%5 != 0: #skip some frames
        continue
        
    frame = cv2.resize(frame, None,fx=0.5, fy=0.5)
    img, factor = yolo.resize_image(frame, yolo.imgsize)
    imgs = yolo.prepare_tensor_image(np.array([img]))
    preds = yolo.predict_images(imgs)
    bboxes = yolo.post_process_predictions(preds)
    bboxes = yolo.select_objects(bboxes, yolo.vehicles)[0]

    plotted = yolo.draw_bbox(frame, bboxes, factor)

    cv2.imshow("prediction", plotted)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
            break
    frames += 1
    # break

