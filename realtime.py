import argparse
import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import webbrowser
import time

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

from ctypes import *
from paddleseg.utils import get_sys_env, logger
from inference import Predictor
import darknet



def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default='./inference_model/deploy.yaml',
        type=str)
    parser.add_argument(
        "--config_file",
        help="The detection config file.",
        default='./configs/yolov4-tiny-use.cfg',
        type=str)
    parser.add_argument(
        "--data_file",
        help="The detection data file.",
        default='./data/obj.data',
        type=str)
    parser.add_argument(
        "--weights",
        help="The detection weights file.",
        default='./best_model/yolov4-tiny-use_best.weights',
        type=str)
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu', 'xpu', 'npu'],
        default="gpu",
        help="Select which device to inference, defaults to cpu.")
    parser.add_argument(
        '--use_trt',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to use Nvidia TensorRT to accelerate prediction.')
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "int8"],
        help='The tensorrt precision.')
    parser.add_argument(
        '--min_subgraph_size',
        default=3,
        type=int,
        help='The min subgraph size in tensorrt prediction.')
    parser.add_argument(
        '--cpu_threads',
        default=10,
        type=int,
        help='Number of threads to predict when using cpu.')
    parser.add_argument(
        '--enable_mkldnn',
        default=False,
        type=eval,
        choices=[True, False],
        help='Enable to use mkldnn to speed up when using cpu.')
    parser.add_argument(
        "--benchmark",
        type=eval,
        default=False,
        help="Whether to log some information about environment, model, configuration and performance.")
    parser.add_argument(
        '--print_detail',
        default=True,
        type=eval,
        choices=[True, False],
        help='Print GLOG information of Paddle Inference.')
    return parser.parse_args()


def calculate_weight(area):
    height = area.shape[0]
    alpha = 2/(height - 1)
    arr = np.zeros(area.shape)
    for i in range(arr.shape[1]):
        arr[:,i] = np.arange(arr.shape[0])
    weight = alpha * arr 
    return weight


def fs_confidence(area, weight):
    walkable_area = area * weight 
    confidence = np.sum(walkable_area)/(area.shape[0]*area.shape[1])
    return confidence


def fd_confidence(area):
    confidence = np.sum(area)/(area.shape[0]*area.shape[1])
    return confidence


def walking_guide(confidence, direction):
    text = None
    index_area = None 
    middle = len(confidence)//2
    n = len(direction)
    diff = (len(confidence) - n)//2
    if confidence[middle] > 0.5:
        text = direction[n//2]
        index_area = middle
    elif np.max(confidence) > 0.2:
        pos = np.argmax(confidence)
        index_area = pos
        if pos < middle - diff:
            text = direction[0]
        elif pos > middle + diff:
            text = direction[n-1]
        else: 
            text = direction[pos-diff]
    else:
        text = 'dead'
    return text, index_area


def env_select(result, index_area, n_areas):
    area_width = int(round(result.shape[1]/n_areas))
    if index_area < 6:
        sel_area = result[:, index_area*area_width:(index_area+1)*area_width]
    else:
        sel_area = result[:, index_area*area_width:]
    sel_area_copy = sel_area.copy()
    sel_area[sel_area != 1] = 0
    sel_area_copy[sel_area_copy != 2] = 0
    area = sel_area + sel_area_copy  
    print(area.shape)
    unique, counts = np.unique(area, return_counts=True)
    a = dict(zip(unique, counts))
    if 0 in a.keys():
        del a[0]
    env = [i for i in a.keys() if a[i] == max(a.values())]
    print(env)
    return env 


def convert2relative(bbox, network):
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)
    bbox_unconverted = (x, y, w, h)
    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)
    bbox_converted = (orig_x, orig_y, orig_width, orig_height)
    return bbox_converted, bbox_unconverted


def clock_angle(mid_x):
    if mid_x < 0.2:
        return "ten"
    if mid_x < 0.4:
        return "eleven"
    if mid_x < 0.6:
        return "twelve"
    if mid_x < 0.8:
        return "one"
    if mid_x < 1.0:
        return "two"
    else:
        return None


def detection(frame, network, class_names, class_colors):
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), 
                               interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
    darknet.print_detections(detections, True)
    darknet.free_image(darknet_image)
    detections_adjusted = []

    detect = False 
    ped_light = []
    for label, confidence, bbox in detections:
        bbox_adjusted, bbox_unconvert = convert2original(frame, bbox)
        x_conv, y_conv, w_conv, h_conv = bbox_adjusted
        x, y, w, h = bbox_unconvert 
        angle = clock_angle(x + w/2)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))
        cv2.putText(frame, angle, (x_conv, y_conv - 10), 0, 0.75, [0, 255, 0], 
                                            thickness=2, lineType=cv2.LINE_AA)
        if label == 'red' or 'green': 
            if x_conv < frame.shape[1]/2:
                de_area = 0
            else:
                de_area = 1 
            if label == 'green':
                color = 1
            else: 
                color = 0 
            ped_light.append(color, de_area)
    image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
    return image, detect, ped_light


def main(args):
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    dec_filter = rs.decimation_filter()
    spat_filter = rs.spatial_filter()
    hole_filter = rs.hole_filling_filter()
    temp_filter = rs.temporal_filter()
    depth_to_disp = rs.disparity_transform(True)
    disp_to_depth = rs.disparity_transform(False)

    colorizer.set_option(rs.option.visual_preset, 1)
    colorizer.set_option(rs.option.color_scheme, 0)
    # dec_filter.set_option(rs.option.filter_magnitude, 1)
    # spat_filter.set_option(rs.option.holes_fill, 2)
    # hole_filter.set_option(rs.option.holes_fill, 2)
    # temp_filter.set_option(rs.option.filter_smooth_alpha, 0.25)
    
    
    logger.info("Input: camera")
    logger.info("Create predictor...")
    predictor = Predictor(args)
    network, class_names, class_colors = darknet.load_network(args.config_file, args.data_file,
                                                            args.weights, batch_size=1)
    logger.info("Start predicting...")

    n_areas = 7
    pre_text = None
    play = None
    file = open('./depth_map/map_1.txt', 'r')
    open_arr = np.loadtxt(file)
    all_fs_confidence = np.zeros((n_areas, 1))
    all_fd_confidence = np.zeros((n_areas, 1))
    pre_env = 0
    index = 0  
    while True:
        frameset = pipeline.wait_for_frames()
        if not frameset:
            break
        frameset = align.process(frameset)
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        depth_frame = dec_filter.process(depth_frame)
        depth_frame = depth_to_disp.process(depth_frame)
        depth_frame = spat_filter.process(depth_frame)
        depth_frame = temp_filter.process(depth_frame)
        depth_frame = disp_to_depth.process(depth_frame)
        depth_frame = hole_filter.process(depth_frame)

        frame = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        pre_time = time.time()
        result, pred_img, add_img = predictor.run(frame, weight=0.6) 
        all_walkable_area = (result==1).astype('int64') + (result==2).astype('int64')
        # print(np.unique(all_walkable_area))
        area_width = int(round(frame.shape[1]/n_areas))
        depth_width = int(round(depth.shape[1]/n_areas))

        depth = depth * depth_scale
        depth[depth < 1.6] = float('-inf')
        depth[depth > 10] = 10
        # open_arr = 10*np.ones(depth.shape)
        depth_open = depth*1/open_arr

        for i in range(n_areas):
            if i < n_areas - 1:
                seg_area = all_walkable_area[:,i*area_width:(i+1)*area_width]
                area_open = depth_open[:,i*depth_width:(i+1)*depth_width]
            else:
                seg_area = all_walkable_area[:,i*area_width:]
                area_open = depth_open[:,i*depth_width:]
            weight = calculate_weight(seg_area)
            all_fs_confidence[i] = fs_confidence(seg_area, weight)
            all_fd_confidence[i] = fd_confidence(area_open)

        all_fd_confidence[all_fd_confidence > 1] = 1 
        all_confidence = np.concatenate((all_fd_confidence, all_fs_confidence),axis=1).min(axis=1)
        print(all_confidence)

        can_go = None
        text, index_area = walking_guide(all_confidence, direction)
        if index_area: 
            env = env_select(result, index_area, n_areas)
            if env != 0 and env != pre_env: 
                cv2.putText(add_img, 'Environment changed', (50, 100), 0, 0.75, [0, 255, 0], 
                                                        thickness=2, lineType=cv2.LINE_AA)
            pre_env = env
            can_go = 1
            
        frame, detect, ped_light = detection(frame, network, class_names, class_colors)     
        if can_go:
            if index_area <= 2: 
                in_area = 0 
            elif index_area >= 4: 
                in_area = 1 
            if env == 2 and detect: 
                for green, de_area in ped_light: 
                    if in_area == de_area or index_area == 3: 
                        if not green: 
                            can_go = 0
        if not can_go: 
            text = 'dead'
            
        play = True
        if text == 'go straight' and text == pre_text:
            play = False
        pre_text = text
        cv2.putText(add_img, text, (50, 50), 0, 0.75, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
        if play:
            webbrowser.open(sounds[text])
        
        fps = 1/(time.time() - pre_time) 
        print('FPS: {}'.format(fps))
        cv2.putText(add_img, 'Sound: ' + str(play), (50, 200), 0, 0.75, [0, 255, 0], 
                                            thickness=2, lineType=cv2.LINE_AA)
        for i in range(n_areas):
            if i > 0 and i < n_areas:
                cv2.line(add_img, (i*area_width,0), (i*area_width,add_img.shape[0]), (0,0,0), 2)
            cv2.putText(add_img, str(round(all_fs_confidence[i][0], 2)), (i*area_width+area_width//3, 390), 0, 0.75, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(add_img, str(round(all_fd_confidence[i][0], 2)), (i*area_width+area_width//3, 420), 0, 0.75, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(add_img, str(round(all_confidence[i], 2)), (i*area_width+area_width//3, 450), 0, 0.75, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
        
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        cv2.imshow('Colorized Depth', colorized_depth)
        cv2.imshow('Real Depth', np.asanyarray(depth_frame.get_data()))
        cv2.imshow('Predict', pred_img)
        cv2.imshow('Frame', frame)
        cv2.imshow('Add', add_img)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            index += 1  
            cv2.imwrite(f'./results/new_{str(index)}.jpg', add_img)
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print(index)
        if index == 20:
            break
    pipeline.stop()

# unlabel = 0 
# sidewalk =  1 
# crosswalk = 2 
# road = 3 

if __name__ == "__main__":
    args = parse_args()
    env_info = get_sys_env()
    args.use_gpu = True if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else False

    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    direction = ['left', 'slightly left', 'go straight', 'slightly right', 'right']
    sounds = {'left': 'sounds/left.mp3', 
              'slightly left': 'sounds/slightly_left.mp3', 
              'go straight': 'sounds/go_straight.mp3', 
              'slightly right': 'sounds/slightly_right.mp3', 
              'right': 'sounds/right.mp3',
              'dead': 'sounds/dead.mp3'}
    main(args)
