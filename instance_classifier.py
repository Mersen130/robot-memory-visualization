import os
from collections import defaultdict 
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

from utilities import parse_box

class YoloClass:
    def __init__(self, clsid2name, ) -> None:
        pass

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

class InstanceClassifier:
    def __init__(self) -> None:
        self.model = models.mobilenet_v2(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[0])
        self.model.eval()
        self.model.to("cuda")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.clsid2name = {}
        self.clsid2concreteID = {}
        self.concreteid2embed = {}
        with open(os.path.join(__location__, 'yolo_classes.txt'), 'r') as file:
            lines = file.readlines()
            for index, line in enumerate(lines):
                name = line.strip()
                self.clsid2name[index] = name
                self.clsid2concreteID[index] = []
        # self.name2clsid = {v: k for k, v in self.clsid2name.items()}

        self.next_id = len(self.clsid2name)

    def get_embedding(self, input_batch):

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        embeddings = None
        with torch.no_grad():
            embeddings = self.model(input_batch)

        # Flatten the embeddings
        embeddings = torch.flatten(embeddings, start_dim=1)

        return embeddings.cpu()
    
    def cosine_similarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def update(self, frame, boxes):
        curr_objs, curr_obj2cls, curr_obj2box = parse_box(boxes)
        
        same_class_objs = defaultdict(list)
        for key, value in curr_obj2cls.items():
            same_class_objs[value].append(key)
        
        obj_mapping = {}
        for clsid, objs in same_class_objs.items():
            if clsid == 0:
                # don't check human
                continue
            
            # get cropped images
            cropped_images = []
            for obj in objs:
                box = curr_obj2box[obj]
                if box[0] < 0:
                    box[0] = 0
                if box[1] < 0:
                    box[1] = 0
                if box[2] < 0:
                    box[2] = 0
                if box[3] < 0:
                    box[3] = 0

                cropped_images.append(self.transform(frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]))
            cropped_images = torch.stack(cropped_images)

            embeddings = self.get_embedding(cropped_images)

            # print(clsid)
            if self.clsid2concreteID[clsid]:  # this class has multiple instances
                for i in range(len(objs)):
                    subject = embeddings[i]
                    for j in range(len(self.clsid2concreteID[clsid])):
                        target = self.concreteid2embed[self.clsid2concreteID[clsid][j]]
                        simi = self.cosine_similarity(subject, target)
                        # print(simi)
                        if simi > 0.5:
                            obj_mapping[objs[i]] = self.clsid2concreteID[clsid][j]
                            break  # TODO: alternative: choose the highest similarity
                    
                    if objs[i] not in obj_mapping:
                        self.clsid2concreteID[clsid].append(self.next_id)  # TODO: cache to local yolo_classes.txt
                        self.clsid2name[self.next_id] = self.clsid2name[clsid] + str(len(self.clsid2concreteID[clsid]))
                        self.concreteid2embed[self.next_id] = subject
                        obj_mapping[objs[i]] = self.next_id
                        self.next_id += 1
            else:
                for i in range(len(objs)):
                    subject = embeddings[i]

                    self.clsid2concreteID[clsid].append(self.next_id)  # cache to local yolo_classes.txt
                    self.clsid2name[self.next_id] = self.clsid2name[clsid] + str(len(self.clsid2concreteID[clsid]))
                    self.concreteid2embed[self.next_id] = subject
                    obj_mapping[objs[i]] = self.next_id
                    self.next_id += 1
            
            # print(self.clsid2concreteID)
            # print(self.clsid2name)
            # print(obj_mapping)

        # modify class ids in boxes
        boxes = self.apply_mapping(boxes, obj_mapping)
        # print(boxes)
        return boxes, self.clsid2name, {v: k for k, v in self.clsid2name.items()}

    
    def apply_mapping(self, boxes, obj_mapping):
        for i in range(len(boxes)):
            if boxes[i][5] != 0:
                obj_id = int(boxes[i][4])
                boxes[i][5] = obj_mapping[obj_id]
        return boxes
        

        
instance_classifier = InstanceClassifier()