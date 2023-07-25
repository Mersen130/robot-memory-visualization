import queue
import threading
import cv2


class Camera:
    def __init__(self, id, width, height):
        self.cam_feed = cv2.VideoCapture(id)
        self.cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self.cam_feed.get(cv2.CAP_PROP_FPS)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cam_feed.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def destroy(self):
        self.cam_feed.release()

    def get_fps(self):
        return self.fps
