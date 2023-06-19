class Flicker_Remover:
    def __init__(self) -> None:
        self.known_objs = set()
        self.last_frame_objs = set()
        self.last_frame_boxes = []
        self.second_last_frame_boxes = []
        self.threshold = 0.5

        self.flickered_objid_mapping = {}

    def update(self, boxes):
        print(self.flickered_objid_mapping)
        self.apply_mapping(boxes)

        curr_objs = set()
        curr_obj2box = {}
        for box in boxes:
            x1, y1, x2, y2, obj_id, cls_id = \
                int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]), int(box[5])
        
            curr_objs.add(obj_id)
            curr_obj2box[obj_id] = box
        
        for obj_id in curr_objs.difference(self.known_objs):  # objs enter
            self.known_objs.add(obj_id)

            similar_box = self.get_similar_boxes(boxes, curr_obj2box[obj_id])
            if similar_box:
                self.flickered_objid_mapping[obj_id] = similar_box
                boxes.remove(curr_obj2box[obj_id])
                continue

            similar_box = self.get_similar_boxes(self.last_frame_boxes, curr_obj2box[obj_id])
            if similar_box:
                self.flickered_objid_mapping[obj_id] = similar_box
                continue

            similar_box = self.get_similar_boxes(self.second_last_frame_boxes, curr_obj2box[obj_id])
            if similar_box:
                self.flickered_objid_mapping[obj_id] = similar_box

        # for obj_id in self.last_frame_objs.difference(curr_objs):  # objs left
        #     pass
        self.apply_mapping(boxes)

        self.last_frame_boxes, self.second_last_frame_boxes = boxes, self.last_frame_boxes
        return boxes
    
    def apply_mapping(self, boxes):
        for i in range(len(boxes)):
            obj_id = int(boxes[i][4])
            if obj_id in self.flickered_objid_mapping:
                boxes[i][4] = self.flickered_objid_mapping[obj_id][4]
                boxes[i][5] = self.flickered_objid_mapping[obj_id][5]
    
    def get_similar_boxes(self, boxes, target):
        """find the box that's similar to target in boxes, they are potentially same object"""
        t_x1, t_y1, t_x2, t_y2, t_obj_id, t_cls_id = \
            int(target[0]), int(target[1]), int(target[2]), int(target[3]), int(target[4]), int(target[5])
        b1 = (t_x1, t_y1, t_x2, t_y2)

        for box in boxes:
            x1, y1, x2, y2, obj_id, cls_id = \
                int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]), int(box[5])
            b2 = (x1, y1, x2, y2)
            if b2 == b1:
                continue

            iou = self.bb_intersection_over_union(b1, b2)
            print("curr", iou)

            if iou >= self.threshold:
                return box
            

        
        return None
    
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

