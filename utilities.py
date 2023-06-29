def parse_box(boxes):
    curr_objs = set()
    curr_obj2cls = {}
    curr_obj2box = {}

    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        obj_id = int(box[4])
        cls_id = int(box[5])
    
        curr_objs.add(obj_id)
        curr_obj2cls[obj_id] = cls_id
        curr_obj2box[obj_id] = box

    return curr_objs, curr_obj2cls, curr_obj2box
