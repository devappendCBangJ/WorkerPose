# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
from collections import namedtuple
import cv2
from pathlib import Path
from FPS import FPS, now
import argparse
import os
from openvino.inference_engine import IENetwork, IECore
import json
import torch
import glob

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models/movenet_singlepose_lightning_FP32.xml"

skeleton_list = []
error_list = []

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# LINES_*_BODY are used when drawing the skeleton onto the source image.
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#keypoint-diagram

LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
              [10,8],[8,6],[6,5],[5,7],[7,9],
              [6,12],[12,11],[11,5],
              [12,14],[14,16],[11,13],[13,15]]

Bbox = []

class Body:
    def __init__(self, scores=None, keypoints_norm=None):
        self.scores = scores  # scores of the keypoints
        self.keypoints_norm = keypoints_norm  # Keypoints normalized ([0,1]) coordinates (x,y) in the squared input image
        self.keypoints = None  # keypoints coordinates (x,y) in pixels in the source image

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))


CropRegion = namedtuple('CropRegion', ['xmin', 'ymin', 'xmax', 'ymax',
                                       'size'])  # All values are in pixel. The region is a square of size 'size' pixels


class MovenetOpenvino:
    def __init__(self, input_src=None,
                 xml=DEFAULT_MODEL,
                 device="CPU",
                 score_thresh=0.2,
                 output=None):
        self.score_thresh = score_thresh
        # if input_src.endswith('.jpg') or input_src.endswith('.png'):
        #     self.input_type = "image"
        #     self.img = cv2.imread(input_src)
        #     self.video_fps = 25
        #     self.img_h, self.img_w = self.img.shape[:2]
        # else:
        #     self.input_type = "video"
        #     if input_src.isdigit():
        #         input_type = "webcam"
        #         input_src = int(input_src)
        #     self.cap = cv2.VideoCapture(input_src)
        #     self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        #     self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print("Video FPS:", self.video_fps)
        self.input_type = "image"
        self.img = input_src
        self.img_h, self.img_w = self.img.shape[:2]

        # Load Openvino models
        self.load_model(xml, device)

        # Rendering flags
        self.show_fps = True
        self.show_crop = False

        if output is None:
            self.output = None
        else:
            if self.input_type == "image":
                # For an source image, we will output one image (and not a video) and exit
                self.output = output
            else:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.output = cv2.VideoWriter(output, fourcc, self.video_fps, (self.img_w, self.img_h))

                # Defines the default crop region (pads the full image from both sides to make it a square image)
        # Used when the algorithm cannot reliably determine the crop region from the previous frame.
        box_size = max(self.img_w, self.img_h)
        x_min = (self.img_w - box_size) // 2
        y_min = (self.img_h - box_size) // 2
        self.init_crop_region = CropRegion(x_min, y_min, x_min + box_size, y_min + box_size, box_size)
        print("init crop", self.init_crop_region)

    def load_model(self, xml_path, device):

        print("Loading Inference Engine")
        self.ie = IECore()
        print("Device info:")
        versions = self.ie.get_versions(device)
        print("{}{}".format(" " * 8, device))
        print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[device].major, versions[device].minor))
        print("{}Build ........... {}".format(" " * 8, versions[device].build_number))

        name = os.path.splitext(xml_path)[0]
        bin_path = name + '.bin'
        print("Pose Detection model - Reading network files:\n\t{}\n\t{}".format(xml_path, bin_path))
        self.pd_net = self.ie.read_network(model=xml_path, weights=bin_path)
        # Input blob: input:0 - shape: [1, 192, 192, 3] (for lightning)
        # Input blob: input:0 - shape: [1, 256, 256, 3] (for thunder)
        # Output blob: 7022.0 - shape: [1, 1, 1]
        # Output blob: 7026.0 - shape: [1, 1, 17]
        # Output blob: Identity - shape: [1, 1, 17, 3]
        self.pd_input_blob = next(iter(self.pd_net.input_info))
        print(
            f"Input blob: {self.pd_input_blob} - shape: {self.pd_net.input_info[self.pd_input_blob].input_data.shape}")
        _, self.pd_h, self.pd_w, _ = self.pd_net.input_info[self.pd_input_blob].input_data.shape
        for o in self.pd_net.outputs.keys():
            print(f"Output blob: {o} - shape: {self.pd_net.outputs[o].shape}")
        self.pd_kps = "Identity"
        print("Loading pose detection model into the plugin")
        self.pd_exec_net = self.ie.load_network(network=self.pd_net, num_requests=1, device_name=device)

        self.infer_nb = 0
        self.infer_time_cumul = 0

    def crop_and_resize(self, frame, crop_region):
        """Crops and resize the image to prepare for the model input."""
        cropped = frame[max(0, crop_region.ymin):min(self.img_h, crop_region.ymax),
                  max(0, crop_region.xmin):min(self.img_w, crop_region.xmax)]

        if crop_region.xmin < 0 or crop_region.xmax >= self.img_w or crop_region.ymin < 0 or crop_region.ymax >= self.img_h:
            # Padding is necessary
            cropped = cv2.copyMakeBorder(cropped,
                                         max(0, -crop_region.ymin),
                                         max(0, crop_region.ymax - self.img_h),
                                         max(0, -crop_region.xmin),
                                         max(0, crop_region.xmax - self.img_w),
                                         cv2.BORDER_CONSTANT)

        cropped = cv2.resize(cropped, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)
        return cropped

    def torso_visible(self, scores):
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((scores[KEYPOINT_DICT['left_hip']] > self.score_thresh or
                 scores[KEYPOINT_DICT['right_hip']] > self.score_thresh) and
                (scores[KEYPOINT_DICT['left_shoulder']] > self.score_thresh or
                 scores[KEYPOINT_DICT['right_shoulder']] > self.score_thresh))

    def determine_torso_and_body_range(self, keypoints, scores, center_x, center_y):
        """Calculates the maximum distance from each keypoints to the center location.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determine_crop_region for more detail.
        """
        # import pdb
        # pdb.set_trace()
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - keypoints[KEYPOINT_DICT[joint]][1])
            dist_x = abs(center_x - keypoints[KEYPOINT_DICT[joint]][0])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for i in range(len(KEYPOINT_DICT)):
            if scores[i] < self.score_thresh:
                continue
            dist_y = abs(center_y - keypoints[i][1])
            dist_x = abs(center_x - keypoints[i][0])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def determine_crop_region(self, body):
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """
        if self.torso_visible(body.scores):
            center_x = (body.keypoints[KEYPOINT_DICT['left_hip']][0] + body.keypoints[KEYPOINT_DICT['right_hip']][
                0]) // 2
            center_y = (body.keypoints[KEYPOINT_DICT['left_hip']][1] + body.keypoints[KEYPOINT_DICT['right_hip']][
                1]) // 2
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = self.determine_torso_and_body_range(
                body.keypoints, body.scores, center_x, center_y)
            crop_length_half = np.amax(
                [max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2])
            tmp = np.array([center_x, self.img_w - center_x, center_y, self.img_h - center_y])
            crop_length_half = int(round(np.amin([crop_length_half, np.amax(tmp)])))
            crop_corner = [center_x - crop_length_half, center_y - crop_length_half]

            if crop_length_half > max(self.img_w, self.img_h) / 2:
                return self.init_crop_region
            else:
                crop_length = crop_length_half * 2
                return CropRegion(crop_corner[0], crop_corner[1], crop_corner[0] + crop_length,
                                  crop_corner[1] + crop_length, crop_length)
        else:
            return self.init_crop_region

    def pd_postprocess(self, inference, crop_region):
        kps = np.squeeze(inference[self.pd_kps])  # 17x3
        # kps = np.where(kps<0, kps+1, kps) # Bug with Openvino 2021.2
        body = Body(scores=kps[:, 2], keypoints_norm=kps[:, [1, 0]])
        body.keypoints = (
                np.array([crop_region.xmin, crop_region.ymin]) + body.keypoints_norm * crop_region.size).astype(
            np.int64)
        body.next_crop_region = self.determine_crop_region(body)
        return body

    def pd_render(self, frame, body, crop_region):

        lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if
                 body.scores[line[0]] > self.score_thresh and body.scores[line[1]] > self.score_thresh]
        #cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)

        for i, x_y in enumerate(body.keypoints):
            skeleton_list.append(int(x_y[0]))
            skeleton_list.append(int(x_y[1]))
            if body.scores[i] > self.score_thresh:
                if i % 2 == 1:
                    color = (0, 255, 0)
                elif i == 0:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

        if self.show_crop:
            cv2.rectangle(frame, (crop_region.xmin, crop_region.ymin), (crop_region.xmax, crop_region.ymax),
                          (0, 255, 255), 2)

        #skeleton_list = body.keypoints.copy()

    def run(self):

        self.fps = FPS()

        nb_pd_inferences = 0
        glob_pd_rtrip_time = 0

        use_previous_keypoints = False

        crop_region = self.init_crop_region

        while True:

            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    break

            cropped = self.crop_and_resize(frame, crop_region)

            frame_nn = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32)[None,]
            pd_rtrip_time = now()
            inference = self.pd_exec_net.infer(inputs={self.pd_input_blob: frame_nn})
            glob_pd_rtrip_time += now() - pd_rtrip_time
            body = self.pd_postprocess(inference, crop_region)
            self.pd_render(frame, body, crop_region)
            crop_region = body.next_crop_region
            nb_pd_inferences += 1

            self.fps.update()

            if self.show_fps:
                self.fps.draw(frame, orig=(50, 50), size=1, color=(240, 180, 100))
            #cv2.imshow("Movepose", frame)

            if self.output:
                if self.input_type == "image":
                    cv2.imwrite(self.output, frame)
                    break
                else:
                    self.output.write(frame)

            #key = cv2.waitKey(0)  # 원래는 1 , 0으로 해서 시작하자마자 멈춤
            break
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('f'):
                self.show_fps = not self.show_fps
            elif key == ord('c'):
                self.show_crop = not self.show_crop

        # Print some stats
        if nb_pd_inferences > 1:
            global_fps, nb_frames = self.fps.get_global()

            print(f"FPS : {global_fps:.1f} f/s (# frames = {nb_frames})")
            print(f"# pose detection inferences : {nb_pd_inferences}")
            print(f"Pose detection round trip   : {glob_pd_rtrip_time / nb_pd_inferences * 1000:.1f} ms")

        if self.output and self.input_type != "image":
            self.output.release()


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='result',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    weights = 'C://Users//user//PycharmProjects//CudaProject//Yolo+Movenet//runs//train//yolov5s_results//weights//best.pt'
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            Bbox.clear()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            Bbox.append(xyxy[0])
            Bbox.append(xyxy[1])
            Bbox.append(xyxy[2])
            Bbox.append(xyxy[3])
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    return im0[int(Bbox[1])-30:int(Bbox[3])+30, int(Bbox[0])-30:int(Bbox[2])+30]
def parse_opt(source_yolo):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=source_yolo)
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    print("opt2 : ", opt)
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print("opt3 : ", opt)
    print_args(vars(opt))
    print("opt4 : ", opt)
    return opt


def main(opt):
    print("opt5 : ", opt)
    check_requirements(exclude=('tensorboard', 'thop'))
    source_dummy = run(**vars(opt))
    print("opt6 : ", opt)
    return source_dummy


if __name__ == "__main__":
    img_list = glob.glob("C://DataSet//images//*.jpg")
    json_list = glob.glob("C://DataSet//Json//*.json")
    for i in range(len(img_list)):
        try:
            skeleton_list.clear()
            with open(json_list[i], "r") as json_file:
                json_data = json.load(json_file)
            source_yolo = str(img_list[i])
            print("opt1 : ", opt)
            opt = parse_opt(source_yolo)
            source_cropped = main(opt)
            parser = argparse.ArgumentParser()
            parser.add_argument('-i', '--input', type=str, default='0',
                                help="Path to video or image file to use as input (default=%(default)s)")
            parser.add_argument("-p", "--precision", type=int, choices=[16, 32], default=32,
                                help="Precision (default=%(default)i")
            parser.add_argument("-m", "--model", type=str, choices=['lightning', 'thunder'], default='thunder',
                                help="Model to use (default=%(default)s")
            parser.add_argument("--xml", type=str,
                                help="Path to an .xml file for model")
            parser.add_argument("-d", "--device", default='CPU', type=str,
                                help="Target device to run the model (default=%(default)s)")
            parser.add_argument("-s", "--score_threshold", default=0.2, type=float,
                                help="Confidence score to determine whether a keypoint prediction is reliable (default=%(default)f)")
            parser.add_argument("-o", "--output",
                                help="Path to output video file")

            args = parser.parse_args()

            if args.device == "MYRIAD":
                args.precision = 16
            if not args.xml:3
                args.xml = SCRIPT_DIR / f"models/movenet_singlepose_{args.model}_FP{args.precision}.xml"

            pd = MovenetOpenvino(input_src=source_cropped,
                                 xml=args.xml,
                                 device=args.device,
                                 score_thresh=args.score_threshold,
                                 output=args.output)
            file_path = "C://DataSet//Test//" + str(json_list[i][-47:])
            pd.run()
            json_data['skeleton'] = (skeleton_list)
            with open(file_path, 'w') as outfile:
                json.dump(json_data, outfile, indent=4)
        except:
            error_list.append(img_list[i])
            continue
    print(error_list)
