import sys
import dlib
from skimage import io
from skimage.transform import rescale
import cv2
import os

#cnn_face_detector = dlib.cnn_face_detection_model_v1('model/face_detector.dat')
cnn_face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/landmark_predictor.dat')


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

    
class Face:
    def __init__(self, bounding_box, conf, folder, tracker=None, face_id=0, num=0, mode='img', scale=(120, 120)):
        self.bb = bounding_box
        self.confidence = conf
        self.tracker = tracker
        self.frame_num = num
        self.id = face_id
        self.still_on_video = True
        self.mode = mode
        self.scale = scale
        self.face_folder = folder + "face_id_{}".format(face_id)
        if not os.path.exists(self.face_folder):
            os.makedirs(self.face_folder)
        if self.mode == 'video':
            #print('yes_1')
            self.video_writer = cv2.VideoWriter(self.face_folder + '/video_{}.avi'.format(face_id),
                                                cv2.VideoWriter_fourcc(*'XVID'), 30.0, self.scale, isColor=False)
    def save_crop(self, img, scale=(120, 120)):
        shape = predictor(img, self.bb)
        center = [0, 0]
        dist = [shape.part(48).x, shape.part(48).x]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(48, 68):
            #print(type(shape.part(i)[0]))
            center[0] += shape.part(i).x
            center[1] += shape.part(i).y
            dist[0] = min(shape.part(i).x, dist[0])
            dist[1] = max(shape.part(i).x, dist[1])
            cv2.circle(gray, (shape.part(i).x, shape.part(i).y), 3, (0,255,0), -1)
        center[0], center[1] = int(center[0]/20), int(center[1]/20)
        distance = dist[1] - dist[0]
        #print(shape.part(0), shape.part(60))
        roi_gray = gray[center[1]-distance:center[1]+distance, center[0]-distance:center[0]+distance]
        roi_gray = cv2.resize(roi_gray, self.scale, interpolation = cv2.INTER_CUBIC)
        #roi_gray = rescale(gray[center[1]-distance:center[1]+distance, center[0]-distance:center[0]+distance], 
        #                   self.scale, mode='reflect')
        if self.mode == 'img':
            io.imsave(self.face_folder + "/frame_{}.png".format(self.frame_num), 
                      roi_gray)
        elif self.mode == 'video':
            #cv2.imshow('lol', roi_gray)
            self.video_writer.write(roi_gray)
        
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def IoU(boxA, boxB):
    intersection = boxA.intersect(boxB).area()
    union = boxA.area() + boxB.area()
    iou = intersection/(union-intersection)
        #print(iou)
    return iou


def detect_faces(image, upsample_time=1):
    faces = []
    dets = cnn_face_detector(image, upsample_time)
    for i, detected_face in enumerate(dets):
        #x1, y1, x2, y2, detector_confidence = detected_face.rect.left(), detected_face.rect.top(), detected_face.rect.right(), #detected_face.rect.bottom(), detected_face.confidence
        x1, y1, x2, y2, detector_confidence = detected_face.left(), detected_face.top(), detected_face.right(), detected_face.bottom(), 1.0
        faces.append((x1, y1, x2, y2, detector_confidence))
    return faces


def detect_on_video(video, upsample_time=1, det_threshold=0.5, inter_threshold=0.5, step=24, faces_folder='faces/', _mode='img'):
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    persons = []
    cur_id = 0
    while(cap.isOpened()):
        progress(frame_id, length, "Extracting mouthes")
        ret, frame = cap.read()
        if(ret):
            if frame_id % step == 0:
                #print(frame_id)
                for existed_face in persons:
                    existed_face.still_on_video = False
                faces = detect_faces(frame[:,:,::-1], upsample_time)
                for face in faces:
                    x1, y1, x2, y2, conf = face
                    if conf > det_threshold:
                        face_bb = dlib.rectangle(x1, y1, x2, y2)
                        #print(face_bb)
                        new_face = True
                        for existed_face in persons:
                            if IoU(existed_face.bb, face_bb) >= inter_threshold:
                                existed_face.bb = face_bb
                                existed_face.tracker.start_track(frame, face_bb)
                                existed_face.still_on_video = True
                                new_face = False
                        if new_face:
                            #print('yes_2')
                            persons.append(Face(face_bb, conf, faces_folder, dlib.correlation_tracker(), cur_id, frame_id, mode=_mode))
                            cur_id += 1
                            persons[-1].tracker.start_track(frame, face_bb)
                for existed_face in persons:
                    #print(existed_face.id)
                    if existed_face.still_on_video == False:
                        #print(existed_face.id)
                        #print()
                        #print(existed_face.bb)
                        if existed_face.mode == 'video':
                            existed_face.video_writer.release()
                        persons.remove(existed_face)
                    else:
                        #print(existed_face.bb, existed_face.id)
                        existed_face.save_crop(frame)
                        existed_face.frame_num += 1
            else:
                for existed_face in persons:
                    existed_face.tracker.update(frame)
                    existed_face.save_crop(frame)
                    existed_face.frame_num += 1
        else:
            break
        frame_id += 1
        
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You don't have one more argument: path to your image")
        exit()
    detect_on_video(sys.argv[1], _mode='video')