import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image as ri
import torchvision
sys.path.append('.')
sys.path.append('./BoT-SORT')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
import configparser

from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs
import detectron2.data.transforms as T

from tracker.tracking_utils.timer import Timer
from tracker.bot_sort import BoTSORT

from diffusiondet import DiffusionDetDatasetMapper#, build_transform_gen

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Tracks For Evaluation!")

    parser.add_argument("path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default='exp1')
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")
    parser.add_argument("--output-dir", default='./')


    
    # Diffusion det args
    parser.add_argument("--config-diffdet-file", default='configs/diffdet.mot.swinbase.yaml')
    parser.add_argument("--config-weights-file", default='output_swin1/model_0002999.pth')
    parser.add_argument("--self-condition", action='store_true', default=False)
    parser.add_argument("--as-video", default='yes')
    parser.add_argument("--time", default=None)
    


    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"BoT-SORT/pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class myDefaultPredictor(DefaultPredictor):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        #self.tfm_gens = build_transform_gen(cfg, False)

    def __call__(self, original_image, previous=None, time=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.

            #if self.input_format == "RGB":
            #    # whether the model expects BGR inputs or RGB
            #    original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)#original_image
            
            #image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}

            
            predictions = self.model([inputs], previous, time)
            
            return predictions

def get_diffdet(args):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.MODEL.DEVICE = 'cuda:2'
    cfg.merge_from_file(args.config_diffdet_file)
    cfg.merge_from_list(['MODEL.WEIGHTS', args.config_weights_file])


    #cfg.DATASETS.TEST = (args.dataset+'_val',)
    #cfg.DATASETS.TRAIN = (args.dataset+'_train',)
    cfg.AS_VIDEO = args.as_video
    cfg.TIME = args.time
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 1
    #cfg.T = None
    cfg.SELF_CONDITION = args.self_condition

    return myDefaultPredictor(cfg)

def image_track(predictor, vis_folder, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    train_ = 0
    if args.split_to_eval=='train':
        train_ = int(len(files)*0.8)
        files = files[:train_]
    elif args.split_to_eval=='val':
        train_ = int(len(files)*0.8)
        files = files[train_:]

    num_frames = len(files) + train_

    path = "/".join(files[0].split('/')[:-2])
    config = configparser.ConfigParser()
    config.read(path + '/seqinfo.ini')
    args.track_buffer = int(config['Sequence']['framerate'])

    # Tracker
    tracker = BoTSORT(args, frame_rate=args.fps)

    results = []
    
    
    frameSize = (int(config['Sequence']['imWidth']), int(config['Sequence']['imHeight']))
    vid = files[0].split('/')[-3]
    output_path = f'{vis_folder}/{vid}.avi'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), int(config['Sequence']['framerate']), frameSize)
    prev = None
    for img_path in files:
        frame_id = int(img_path.split('/')[-1].split('.')[0])
        # Detect objects
        img = read_image(img_path, format='RGB')
        timer.tic()
        if (not args.self_condition and args.time is None) or prev is None:
            outputs, prev = predictor(img)
            
        else:
            outputs, prev = predictor(img, prev, [[args.time]])

        boxes = outputs[0]['instances'].get_fields()['pred_boxes'].tensor
        scores = outputs[0]['instances'].get_fields()['scores'][:, None]
        classes = outputs[0]['instances'].get_fields()['pred_classes'][:, None]
        boxes = boxes[scores[:,0]>0.05]
        classes = classes[scores[:,0]>0.05]
        scores = scores[scores[:,0]>0.05]
        

        #scores[scores[:,0]>0.1] = ((scores[scores[:,0]>0.1] - scores[scores[:,0]>0.1].min()) / (scores[scores[:,0]>0.1].max() - scores[scores[:,0]>0.1].min()))/2 + 0.5
        #scores[scores[:,0]<0.1] = ((scores[scores[:,0]<0.1]) / (scores[scores[:,0]<0.1].max()))/2 
        
        outputs = torch.cat((boxes, scores, classes), 1)
        
        #print(outputs[outputs[:,-2]>0.05].shape, frame_id)


        outputs = [outputs[outputs[:,-1] == 0]]
        
        img_info = {}
        img_info['raw_img'] = img
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["file_name"] = img_path
        
        
        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]

            trackerTimer.tic()
            online_targets = tracker.update(detections, img_info["raw_img"])
            trackerTimer.toc()

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
            out.write(online_im)
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if args.save_frames:
            save_folder = osp.join(vis_folder, args.name)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))
    
    out.release()
    res_file = osp.join(vis_folder, args.name + ".txt")

    with open(res_file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {res_file}")


def main(args):

    output_dir = osp.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, args.split_to_eval)
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda:2" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    model = get_diffdet(args)
    #logger.info("Model Summary: {}".format(get_model_info(model)))

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    

    image_track(model, vis_folder, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    data_path = args.path
    fp16 = args.fp16
    device = args.device

    if args.benchmark == 'MOT20':
        train_seqs = [1, 2, 3, 5]
        test_seqs = [4, 6, 7, 8]
        seqs_ext = ['']
        MOT = 20
    elif args.benchmark == 'MOT17':
        train_seqs = [2, 4, 5, 9, 10, 11, 13]
        test_seqs = [1, 3, 6, 7, 8, 12, 14]
        seqs_ext = ['FRCNN']
        MOT = 17
    else:
        raise ValueError("Error: Unsupported benchmark:" + args.benchmark)

    
    if args.split_to_eval == 'train':
        seqs = train_seqs
    elif args.split_to_eval == 'val':
        seqs = train_seqs
    elif args.split_to_eval == 'test':
        seqs = test_seqs
    else:
        raise ValueError("Error: Unsupported split to evaluate:" + args.split_to_eval)

    mainTimer = Timer()
    mainTimer.tic()

    for ext in seqs_ext:
        for i in seqs:
            if i < 10:
                seq = 'MOT' + str(MOT) + '-0' + str(i)
            else:
                seq = 'MOT' + str(MOT) + '-' + str(i)

            if ext != '':
                seq += '-' + ext

            args.name = seq
            args.ablation = True
            args.mot20 = MOT == 20
            args.fps = 30
            args.device = device
            args.fp16 = fp16
            args.batch_size = 1
            args.trt = False

            split = 'train' if i in train_seqs else 'test'
            args.path = data_path + '/' + split + '/' + seq + '/' + 'img1'

            if args.default_parameters:
                args.track_high_thresh = 0.5
                args.track_low_thresh = 0.05
                args.track_buffer = 30

                if seq == 'MOT17-05-FRCNN' or seq == 'MOT17-06-FRCNN':
                    args.track_buffer = 14
                elif seq == 'MOT17-13-FRCNN' or seq == 'MOT17-14-FRCNN':
                    args.track_buffer = 25
                else:
                    args.track_buffer = 30

                if seq == 'MOT17-02-FRCNN' or seq == 'MOT17-09-FRCNN' or seq == 'MOT17-11-FRCNN' or seq == 'MOT17-13-FRCNN':
                    args.track_high_thresh = 0.6#0.65
                #elif seq == 'MOT17-06-FRCNN':
                #    args.track_high_thresh = 0.1#0.65
                #elif seq == 'MOT17-12-FRCNN':
                #    args.track_high_thresh = 0.1#0.7
                #elif seq == 'MOT17-14-FRCNN':
                #    args.track_high_thresh = 0.1#0.67
                #elif seq in ['MOT20-06', 'MOT20-08']:
                #    args.track_high_thresh = 0.1

                args.new_track_thresh = args.track_high_thresh + 0.1
            
            main(args)
            
            

    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 /timer.average_time))
    print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))

