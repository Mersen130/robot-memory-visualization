import cv2
from collections import deque
import os
import threading


class Recorder:
    """
    Record a video when new object appears and left
    """
    def __init__(self, fps, width, height) -> None:
        self.known_objs = set()
        self.last_frame_objs = set()
        self.curr_videowriter = []
        self.frame_dim = (width, height)

        self.fps = fps
        self._3_sec_frame_count = int(fps*3)
        self._3_sec_frames = deque(maxlen=self._3_sec_frame_count)
        self._3_sec_frames_boxes = deque(maxlen=self._3_sec_frame_count)
        self.recording_dir = self.make_newdir()
        self.saver_threads = []

    def update(self, frame, boxes):
        self._3_sec_frames.append(frame)
        self.update_writer()

        curr_objs = set()
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            obj_id = int(box[4])
            cls_id = int(box[5])
        
            curr_objs.add(obj_id)
        
        for obj_id in curr_objs.difference(self.known_objs):  # objs enter
            self.known_objs.add(obj_id)
            self.create_writer("id:{}_entry.mp4".format(obj_id), self._3_sec_frame_count-1)
        
        for obj_id in self.last_frame_objs.difference(curr_objs):  # objs left
            self.create_writer("id:{}_left.mp4".format(obj_id), 0)

        self.last_frame_objs = curr_objs
        self._3_sec_frames_boxes.append(boxes)

    def create_writer(self, filename, frame_to_write):
        writer = cv2.VideoWriter(
                    os.path.join(self.recording_dir, filename), 
                    cv2.VideoWriter_fourcc(*'MP4V'), self.fps, self.frame_dim)
        self.curr_videowriter.append([frame_to_write, writer])
    
    def update_writer(self):
        """update countdown timer for each writer, dispatch them if countdown is 0"""
        curr_videowriter_cpy = self.curr_videowriter[:]
        for i in range(len(curr_videowriter_cpy)):
            curr_videowriter_cpy[i][0] -= 1

            if curr_videowriter_cpy[i][0] == 0:

                t = threading.Thread(target=self.dispatch_writer, 
                                     args=((curr_videowriter_cpy[i][1], 
                                            self._3_sec_frames.copy())))
                t.start()
                self.curr_videowriter.remove(curr_videowriter_cpy[i])
    
    def dispatch_writer(self, writer, frames):
        """write cached frames to video files"""
        for frame in frames:
            writer.write(frame)
        writer.release()

    def destroy(self):
        """disatch all writers"""
        for count_down, writer in self.curr_videowriter:
            t = threading.Thread(target=self.dispatch_writer, args=((writer, self._3_sec_frames)))
            t.start()
            self.saver_threads.append(t)
        
        for t in self.saver_threads:
            t.join()

    def make_newdir(self):
        """create dir for storing recordings"""
        recording_dir = os.path.join(os.getcwd(), 'recordings')
        if not os.path.exists(recording_dir):
            os.mkdir(recording_dir)

        folder_names = os.listdir(os.path.join(os.getcwd(), 'recordings'))
        names = [int(name) for name in folder_names if name.isnumeric()] or [-1]
        last_number = max(names)
        new_name = os.path.join(os.getcwd(), 'recordings', str(last_number + 1))
        os.mkdir(new_name)
        return new_name