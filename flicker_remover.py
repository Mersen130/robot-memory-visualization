class Flicker_Remover:
    def __init__(self) -> None:
        self.known_objs = set()
        self.last_frame_objs = set()
        self.last_frame_boxes = []
        self.second_last_frame_boxes = []

        self.flickered_objid_mapping = {}

    def update(self, boxes):
        self.apply_mapping(boxes)
        # print(self.flickered_objid_mapping)

        curr_objs = set()
        curr_obj2box = {}
        for box in boxes:
            x1, y1, x2, y2, obj_id, cls_id = \
                int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]), int(box[5])
        
            curr_objs.add(obj_id)
            curr_obj2box[obj_id] = box
        
        for obj_id in curr_objs.difference(self.known_objs):  # objs enter
            self.known_objs.add(obj_id)
            similar_box = self.get_similar_boxes(self.last_frame_boxes, curr_obj2box[obj_id])
            if similar_box:
                self.flickered_objid_mapping[obj_id] = int(similar_box[4])
                continue

            similar_box = self.get_similar_boxes(self.second_last_frame_boxes, curr_obj2box[obj_id])
            if similar_box:
                self.flickered_objid_mapping[obj_id] = int(similar_box[4])

        for obj_id in self.last_frame_objs.difference(curr_objs):  # objs left
            pass

        self.last_frame_boxes, self.second_last_frame_boxes = boxes, self.last_frame_boxes
        return boxes
    
    def apply_mapping(self, boxes):
        for box in boxes:
            box[4] = self.flickered_objid_mapping.get(int(box[4]), box[4])
    
    def get_similar_boxes(self, boxes, target):
        """find the box that's similar to target in boxes, they are potentially same object"""
        t_x1, t_y1, t_x2, t_y2, t_obj_id, t_cls_id = \
            int(target[0]), int(target[1]), int(target[2]), int(target[3]), int(target[4]), int(target[5])

        for box in boxes:
            x1, y1, x2, y2, obj_id, cls_id = \
                int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]), int(box[5])

            if t_x1 == x1 and t_y1 == y1 and t_x2 == x2 and t_y2 == y2:
                return box
        
        return None

