import cv2
from collections import deque
import os
import threading
import datetime

CLASS_PERSON = 0

class Frame:
    curr_objs: set
    curr_obj2cls: dict

    def __init__(self, frame, curr_objs, curr_obj2cls) -> None:
        self.frame = frame
        self.curr_objs = curr_objs
        self.curr_obj2cls = curr_obj2cls

class Recorder:
    """
    Record a video when new object appears and left
    """
    def __init__(self, fps, width, height, yolo_id2name) -> None:
        self.known_objs = set()
        self.last_frame_objs = set()
        self.last_frame_obj2cls = {}
        self.curr_videowriter = []
        self.frame_dim = (width, height)
        self.human_event = False
        self.yolo_id2name = yolo_id2name

        self.fps = int(fps)
        self._3_sec_frame_count = int(fps*3)
        self._3_sec_frames = deque(maxlen=self._3_sec_frame_count)
        self._3_sec_frames_objs = deque(maxlen=self._3_sec_frame_count)
        self._1_sec_frames_before_human = []
        self._1_sec_frames_after_human = []
        self._1_sec_frames = deque(maxlen=self.fps)
        self._1_sec_frames_objs = deque(maxlen=self.fps)
        self._1_sec_frames_objs2cls = deque(maxlen=self.fps)

        self.id2filenames = {}
        self.files_to_remove = []
        self._human_event_countdown = 0
        self.human_frames = deque()
        self.objs_before_human = None
        self.objs_during_human = []
        self.obj2cls_before_human = None
        self.recording_dir = self.make_newdir()
        self.saver_threads = []

    def update(self, frame, boxes):
        curr_objs = set()
        curr_obj2cls = {}
        contains_human = False

        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            obj_id = int(box[4])
            cls_id = int(box[5])
            if cls_id == CLASS_PERSON:
                contains_human = True
        
            curr_objs.add(obj_id)
            curr_obj2cls[obj_id] = cls_id
        
        frame_info = Frame(frame, curr_objs, curr_obj2cls)

        self.prepare(frame_info)
        self.update_writer(self._3_sec_frames_objs)

        self.prepare_human_event(frame_info)

        if self.human_event:
            if contains_human:
                self.continue_human_event(frame_info)
            else:
                # human left scene
                self._human_event_countdown = self.fps - 1
                self.stop_human_event(frame_info)
            return
        else:
            if contains_human and not self._human_event_countdown:
                # human enters scene and previous human job finishes
                self.start_human_event(frame)
            else:
                if self._human_event_countdown:
                    self._human_event_countdown -= 1
                    self.stop_human_event(frame_info)
                else:
                    self.start_regular_event(frame_info)
        
        self.finish_update(frame_info)


    def prepare(self, frame_info: Frame):
        self._3_sec_frames.append(frame_info.frame)
        self._3_sec_frames_objs.append(frame_info.curr_objs)

    def finish_update(self, frame_info: Frame):
        self.last_frame_objs = frame_info.curr_objs
        self.last_frame_obj2cls = frame_info.curr_obj2cls

    def start_regular_event(self, frame_info: Frame):
        self.human_event = False
        for obj_id in frame_info.curr_objs.difference(self.known_objs):  # objs enter
            self.known_objs.add(obj_id)
            filename = "{} {} id:{} enter.mp4".format(datetime.datetime.now().strftime("%H:%M:%S"), 
                                                      self.yolo_id2name[frame_info.curr_obj2cls[obj_id]], obj_id)
            
            self.id2filenames[str(obj_id) + "enter"] = filename
            self.create_writer(filename, self._3_sec_frame_count-1, obj_id)
        
        for obj_id in self.last_frame_objs.difference(frame_info.curr_objs):  # objs left
            filename = "{} {} id:{} left.mp4".format(datetime.datetime.now().strftime("%H:%M:%S"), 
                                                     self.yolo_id2name[self.last_frame_obj2cls[obj_id]], obj_id)
            
            self.id2filenames[str(obj_id) + "left"] = filename
            self.create_writer(filename, 1, obj_id)
    
    def prepare_human_event(self, frame_info: Frame):
        self._1_sec_frames_objs.append(frame_info.curr_objs)
        self._1_sec_frames_objs2cls.append(frame_info.curr_obj2cls)
        self._1_sec_frames.append(frame_info.frame)
    
    def start_human_event(self, frame_info: Frame):
        self.human_event = True
        self.human_frames.append(frame_info.frame)
        self.objs_before_human = self._1_sec_frames_objs.copy()
        self.obj2cls_before_human = self._1_sec_frames_objs2cls.copy()
        self._1_sec_frames_before_human = self._1_sec_frames.copy()

        for objs in self._1_sec_frames_objs:
            for obj_id in objs:
                if str(obj_id) + "enter" in self.id2filenames:
                    self.files_to_remove.append(self.id2filenames[str(obj_id) + "enter"])
                if str(obj_id) + "left" in self.id2filenames:
                    self.files_to_remove.append(self.id2filenames[str(obj_id) + "left"])
        
    def continue_human_event(self, frame_info: Frame):
        self.human_frames.append(frame_info.frame)
        self.objs_during_human(frame_info.curr_objs)

    def stop_human_event(self, curr_objs, curr_obj2cls, frame=None):
        self.human_event = False
        if self._human_event_countdown:
            self._1_sec_frames_after_human.append(frame)
            return
        
        frames = deque()
        frames.extend(self._1_sec_frames_before_human)
        frames.extend(self.human_frames)
        frames.extend(self._1_sec_frames_after_human)

        objs = deque()
        objs.extend(self.objs_before_human)
        objs.extend(self.objs_during_human)
        self._1_sec_frames_after_human = []
        self._1_sec_frames_before_human = []
        self.objs_during_human.clear()

        for obj_id in curr_objs.difference(self.objs_before_human[0]):
            self.known_objs.add(obj_id)
            if curr_obj2cls[obj_id] == 0: continue
            filename = "{} {} id:{} enter H.mp4".format(datetime.datetime.now().strftime("%H:%M:%S"),
                                                        self.yolo_id2name[curr_obj2cls[obj_id]], obj_id)
            # self.id2filenames[obj_id] = filename
            t = threading.Thread(target=self.dispatch_writer, 
                                     args=((filename, 
                                            frames, objs, obj_id)))
            t.start()
        
        for obj_id in self.objs_before_human[0].difference(curr_objs):
            if self.obj2cls_before_human[0][obj_id] == 0: continue
            filename = "{} {} id:{} left H.mp4".format(datetime.datetime.now().strftime("%H:%M:%S"),
                                                       self.yolo_id2name[self.obj2cls_before_human[0][obj_id]], obj_id)
            # self.id2filenames[obj_id] = filename
            t = threading.Thread(target=self.dispatch_writer, 
                                     args=((filename, 
                                            frames, objs, obj_id)))
            t.start()
            
        self.human_frames.clear()

    def create_writer(self, filename, frame_to_write, obj_id):
        self.curr_videowriter.append([frame_to_write, filename, obj_id])
    
    def update_writer(self, objs):
        """update countdown timer for each writer, dispatch them if countdown is 0"""
        curr_videowriter_cpy = self.curr_videowriter[:]
        objs = objs.copy()
        frames = self._3_sec_frames.copy()
        for i in range(len(curr_videowriter_cpy)):
            curr_videowriter_cpy[i][0] -= 1

            if curr_videowriter_cpy[i][0] == 0:

                t = threading.Thread(target=self.dispatch_writer, 
                                     args=((curr_videowriter_cpy[i][1], 
                                            frames,
                                            objs, curr_videowriter_cpy[i][2])))
                t.start()
                self.curr_videowriter.remove(curr_videowriter_cpy[i])

    
    def dispatch_writer(self, filename, frames, objs, obj_id):
        """write cached frames to video files, drop recording if obj_id appeared in less than 5 frames"""
        count = 0
        for _objs in objs:
            if obj_id in _objs:
                count += 1
                if count == 5:
                    break
        
        if count < 5:
            return  # this recording is probably caused by a glitch
        
        writer = cv2.VideoWriter(
                    os.path.join(self.recording_dir, filename), 
                    cv2.VideoWriter_fourcc(*'MP4V'), self.fps, self.frame_dim)
        for frame in frames:
            writer.write(frame)
        writer.release()

    def destroy(self):
        """dispatch all writers and do cleaning"""
        for count_down, writer, obj_id in self.curr_videowriter:
            t = threading.Thread(target=self.dispatch_writer, args=((writer, self._3_sec_frames, 
                                                                     self._3_sec_frames_objs, obj_id)))
            t.start()
            self.saver_threads.append(t)
        
        for t in self.saver_threads:
            t.join()

        self.remove_files()
        
    def remove_files(self):
        for file in self.files_to_remove:
            file = os.path.join(self.recording_dir, file)
            if os.path.isfile(file):
                os.remove(file)
                print("cleaning", file)

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
