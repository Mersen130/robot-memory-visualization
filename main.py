import torch
import cv2
from camera_module import Camera

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom


width = 640
height = 480
camera = Camera(-1, width, height)

frames = 0
while True:
	img = camera.read()

	results = model(img)

	print(results)

	for (x,y,w,h,confidence,label) in results.xyxy[0]:
		x,y,w,h,confidence,label = int(x), int(y), int(w), int(h), float(confidence), int(label)
		cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
	
	cv2.imshow("image", img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows() 
camera.destroy()