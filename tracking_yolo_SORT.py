import torch
import cv2
from camera_module import Camera
from Sort import *

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
mot_tracker = Sort()

width = 640
height = 480
camera = Camera(-1, width, height)
colours = np.random.rand(32, 3) #used only for display


id2name = {}
with open('yolo_classes.txt', 'r') as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        name = line.strip()
        id2name[index] = name
	

frames = 0
while True:
	img = camera.read()

	results = model(img)
	detections = results.pred[0].cpu().numpy()

	track_bbs_ids = mot_tracker.update(detections)

	print(results)

	for i in range(len(track_bbs_ids.tolist())):
		coords = track_bbs_ids.tolist()[i]
		x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
		obj_id = int(coords[4])
		cls_id = int(coords[5])
		# print(cls_id)
		name = "{}: {}".format(id2name[cls_id], obj_id)
		color = colours[obj_id % len(colours)] * 255
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
		cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
		
	cv2.imshow("image", img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows() 
camera.destroy()