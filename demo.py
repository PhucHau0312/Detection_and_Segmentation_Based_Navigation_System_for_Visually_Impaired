import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))
from paddleseg.utils import get_sys_env, logger
from inference import Predictor
from PIL import Image 



def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default='./inference_model/deploy.yaml',
        type=str)
    parser.add_argument(
        '--image_path',
        dest='img_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default=None)
    parser.add_argument(
        '--video_path', help='Video path for inference', type=str)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)
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
        help="Whether to log some information about environment, model, configuration and performance."
    )
    parser.add_argument(
        '--print_detail',
        default=True,
        type=eval,
        choices=[True, False],
        help='Print GLOG information of Paddle Inference.')
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output/')

    return parser.parse_args()


def calculate_weight(area):
    height = area.shape[0]
    alpha = 2/(height - 1)
    arr = np.zeros(area.shape)
    for i in range(arr.shape[1]):
        arr[:,i] = np.arange(arr.shape[0])
    weight = alpha * arr 
    return weight


def makedirs(save_dir):
    dirname = save_dir if os.path.isdir(save_dir) else \
        os.path.dirname(save_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
 

def fs_confidence(area, weight):
    walkable_area = area * weight 
    fs_confidence = np.sum(walkable_area)/(area.shape[0]*area.shape[1])
    return fs_confidence


def fd_confidence():
    fd_confidence = []
    return fd_confidence


def walking_guide(confidence, direction):
    middle = len(confidence)//2
    n = len(direction)
    diff = (len(confidence) - n)//2
    if confidence[middle] > 0.5:
        text = direction[n//2]
    elif np.max(confidence) > 0.2:
        pos = np.argmax(confidence)
        if pos < middle - diff:
            text = direction[0]
        elif pos > middle + diff:
            text = direction[n-1]
        else: 
            text = direction[pos-diff]
    return text


def seg_image(args):
    assert os.path.exists(args.img_path), \
        "The --img_path is not existed: {}.".format(args.img_path)

    logger.info("Input: image")
    logger.info("Create predictor...")
    predictor = Predictor(args)
    logger.info("Start predicting...")

    n_areas = 7
    pre_text = None
    play = None
    all_fs_confidence = np.zeros((n_areas,))
    all_fd_confidence = np.zeros((n_areas,))

    img = cv2.imread(args.img_path)
    result, pred_img, add_img = predictor.run(img, weight=0.6)
    all_walkable_area = (result==1).astype('int64') + (result==2).astype('int64')
    print(np.unique(all_walkable_area))
    area_width = int(round(all_walkable_area.shape[1]/n_areas))
    print(area_width)
    for i in range(n_areas):
        if i < n_areas - 1:
            area = all_walkable_area[:,i*area_width:(i+1)*area_width]
        else:
            area = all_walkable_area[:,i*area_width:]

        weight = calculate_weight(area)
        all_fs_confidence[i] = fs_confidence(area, weight)

    print(all_fs_confidence)
    # all_confidence = np.concatenate((all_fd_confidence, all_fs_confidence),axis=0).min(axis=0)
    # text = walking_guide(all_confidence)
    text = walking_guide(all_fs_confidence, direction)
    if text == 'go straight' and text == pre_text:
        play = False
    print(play)
    pre_text = text 
    cv2.putText(add_img, text, (50, 100), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)

    for i in range(n_areas):
        if i > 0 and i < n_areas:
            cv2.line(add_img, (i*area_width,0), (i*area_width,add_img.shape[0]), (0,0,0), 2)
        cv2.putText(add_img, str(round(all_fs_confidence[i], 2)), (i*area_width+area_width//3, 600), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('Result', add_img)
    cv2.waitKey(0)
    cv2.imwrite(args.save_dir + args.img_path, add_img)


def seg_video(args):
    assert os.path.exists(args.video_path), \
        'The --video_path is not existed: {}'.format(args.video_path)
    assert args.save_dir.endswith(".avi"), 'The --save_dir should be xxx.avi'

    cap_img = cv2.VideoCapture(args.video_path)
    assert cap_img.isOpened(), "Fail to open video:{}".format(args.video_path)
    fps = cap_img.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_img.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_img.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_img.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out = cv2.VideoWriter(args.save_dir,
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (width, height))
    logger.info("Input: video")
    logger.info("Create predictor...")
    predictor = Predictor(args)
    logger.info("Start predicting...")
    
    with tqdm(total=total_frames) as pbar:
        img_frame_idx = 0
        n_areas = 7
        pre_text = None
        play = None
        all_fs_confidence = np.zeros((n_areas,))
        all_fd_confidence = np.zeros((n_areas,))
        while cap_img.isOpened():
            ret_img, img = cap_img.read()
            if not ret_img:
                break
            result, pred_img, add_img = predictor.run(img, weight=0.6)
            
            all_walkable_area = (result==1).astype('int64') + (result==2).astype('int64')
            print(np.unique(all_walkable_area))
            area_width = int(round(all_walkable_area.shape[1]/n_areas))

            for i in range(n_areas):
                if i < n_areas - 1:
                    area = all_walkable_area[:,i*area_width:(i+1)*area_width]
                else:
                    area = all_walkable_area[:,i*area_width:]

                weight = calculate_weight(area)
                all_fs_confidence[i] = fs_confidence(area, weight)

            print(all_fs_confidence)
            # all_confidence = np.concatenate((all_fd_confidence, all_fs_confidence),axis=0).min(axis=0)
            # text = walking_guide(all_confidence)
            text = walking_guide(all_fs_confidence, direction)
            if text == 'go straight' and text == pre_text:
                play = False
            print(play)
            pre_text = text 
            cv2.putText(add_img, text, (50, 100), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)

            for i in range(n_areas):
                if i > 0 and i < n_areas:
                    cv2.line(add_img, (i*area_width,0), (i*area_width,add_img.shape[0]), (0,0,0), 2)
                cv2.putText(add_img, str(round(all_fs_confidence[i], 2)), (i*area_width+area_width//3, 600), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            
            cap_out.write(add_img)
            img_frame_idx += 1
            pbar.update(1)
    cap_img.release()
    cap_out.release()


def seg_camera(args):
    cap_camera = cv2.VideoCapture(0)
    assert cap_camera.isOpened(), "Fail to open camera"
    logger.info("Input: camera")
    logger.info("Create predictor...")
    predictor = Predictor(args)
    logger.info("Start predicting...")

    n_areas = 7
    pre_text = None
    play = None
    all_fs_confidence = np.zeros((n_areas,))
    all_fd_confidence = np.zeros((n_areas,))
    while cap_camera.isOpened():
        ret_img, img = cap_camera.read()
        if not ret_img:
            break
        result, pred_img, add_img = predictor.run(img, weight=0.6) 
        
        all_walkable_area = (result==1).astype('int64') + (result==2).astype('int64')
        print(np.unique(all_walkable_area))
        area_width = int(round(all_walkable_area.shape[1]/n_areas))

        for i in range(n_areas):
            if i < n_areas - 1:
                area = all_walkable_area[:,i*area_width:(i+1)*area_width]
            else:
                area = all_walkable_area[:,i*area_width:]

            weight = calculate_weight(area)
            all_fs_confidence[i] = fs_confidence(area, weight)

        print(all_fs_confidence)
        # all_confidence = np.concatenate((all_fd_confidence, all_fs_confidence),axis=0).min(axis=0)
        # text = walking_guide(all_confidence)
        all_confidence = np.concatenate((all_fd_confidence, all_fs_confidence),axis=0).min(axis=0)
        text = walking_guide(all_fs_confidence, direction)
        if text == 'go straight' and text == pre_text:
            play = False
        print(play)
        pre_text = text
        cv2.putText(add_img, text, (50, 100), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)

        for i in range(n_areas):
            if i > 0 and i < n_areas:
                cv2.line(add_img, (i*area_width,0), (i*area_width,add_img.shape[0]), (0,0,0), 2)
            cv2.putText(add_img, str(all_fs_confidence[i]), (i*area_width+area_width//3, 1000), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            
        cv2.imshow('Add', add_img)
        cv2.imshow('Predict', pred_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap_camera.release()

# unlabel = 0 
# sidewalk =  1 
# crosswalk = 2 
# road = 3 

if __name__ == "__main__":
    args = parse_args()
    env_info = get_sys_env()
    args.use_gpu = True if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else False
    makedirs(args.save_dir)

    direction = ['left', 'slightly left', 'go straight', 'slightly right', 'right']
    if args.img_path is not None:
        seg_image(args)
    elif args.video_path is not None:
        seg_video(args)
    else:
        seg_camera(args)
