import torch
import cv2
from camera_module import Camera
from recording_module import Recorder
from flicker_remover import flicker_remover
from Sort import *


def main():
	record_all = True

	__location__ = os.path.realpath(
		os.path.join(os.getcwd(), os.path.dirname(__file__)))

	model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
	mot_tracker = Sort()

	id2name = {}
	with open(os.path.join(__location__, 'yolo_classes.txt'), 'r') as file:
		lines = file.readlines()
		for index, line in enumerate(lines):
			name = line.strip()
			id2name[index] = name

	width = 640
	height = 480
	camera = Camera(-1, width, height)
	recorder = Recorder(camera.get_fps(), width, height, id2name)
	colours = np.random.rand(32, 3) #used only for display


	frames = []
	while True:
		frame = camera.read()

		results = model(frame)
		detections = results.pred[0].cpu().numpy()

		track_bbs_ids = mot_tracker.update(detections).tolist()
		track_bbs_ids = flicker_remover.update(track_bbs_ids)

		# print(results)

		for i in range(len(track_bbs_ids)):
			box = track_bbs_ids[i]
			x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
			obj_id = int(box[4])
			cls_id = int(box[5])
			# print(cls_id)
			name = "{}: {}".format(id2name[cls_id], obj_id)
			color = colours[obj_id % len(colours)] * 255
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
			cv2.putText(frame, name, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
			
		recorder.update(frame, track_bbs_ids)

		cv2.imshow("image", frame)

		frames.append(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	if record_all:
		writer = cv2.VideoWriter("recordings/recording.mp4", 
				cv2.VideoWriter_fourcc(*'MP4V'), camera.get_fps(), (width, height))
		for frame in frames:
			writer.write(frame)
		writer.release()

	cv2.destroyAllWindows() 
	camera.destroy()
	recorder.destroy()

if __name__ == "__main__":
	main()