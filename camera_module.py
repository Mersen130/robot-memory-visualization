import cv2


class Camera:
    def __init__(self, id, width, height):
        self.cam_feed = cv2.VideoCapture(id)
        self.cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ret, img = self.cam_feed.read()
        return img

    def destroy(self):
        self.cam_feed.release()

