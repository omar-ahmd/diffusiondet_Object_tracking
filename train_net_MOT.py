# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import json
import copy
import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict, abc
from tqdm import tqdm
import torch
from torch import nn
from fvcore.nn.precise_bn import get_bn_modules
import glob 
from tabulate import tabulate

from contextlib import ExitStack
import configparser
import pandas as pd
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import (COCOEvaluator, 
                                    LVISEvaluator, 
                                    verify_results, 
                                    DatasetEvaluator, 
                                    DatasetEvaluators)
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import MetadataCatalog
from detectron2.config import CfgNode
from detectron2.utils.file_io import PathManager
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from  detectron2.evaluation import inference_context

from  detectron2.structures import BoxMode

from os.path import exists
import torch.utils.data as torchdata
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer


class MOTEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
    ):
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        self._use_fast_impl = use_fast_impl

        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]
        self._max_dets_per_image = max_dets_per_image

        self._cpu_device = torch.device("cpu")
        self.dataset_name = dataset_name
        
        self._metadata = MetadataCatalog.get('coco_2017_val')
        
        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
        video = dataset_name.split('MOT17_val_')[1].split('_')[0]
        try:
            time = dataset_name.split('MOT17_val_')[1].split('_')[1]
        except:
            time = None
        self.config = configparser.ConfigParser()
        try:
            self.config.read(f'datasets/MOT17/train/MOT17-{video}-DPM/seqinfo.ini')
        except:
            try:
                self.config.read(f'datasets/MOT17/test/MOT17-{video}-DPM/seqinfo.ini')
            except:
                raise NotADirectoryError(f'datasets/MOT17/test/MOT17-{video}-DPM/seqinfo.ini does not exist' )

        
        gt = pd.read_csv(f'datasets/MOT17/train/MOT17-{video}-DPM/gt/gt.txt', header=None)
        
        gt = gt.rename(columns={0:'image_id', 1:'object_id'})
        gt[4] = gt[4] + gt[2]
        gt[5] = gt[5] + gt[3]

        gt[4] = gt[4].clip(0,int(self.config['Sequence']['imWidth']))
        gt[5] = gt[5].clip(0,int(self.config['Sequence']['imHeight']))
        gt = gt[gt[8]>=0.3] #Take only the boxes that have a visibility grater than 0.3
        self.gt = gt[gt[7]==1]
        self.results = None

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None, time=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "instances" in predictions[0]:
            mAP = self._eval_predictions(predictions, img_ids=img_ids, time=time)
            
        # Copy so the caller can do whatever with results
        return mAP, copy.deepcopy(self._results)

    def _eval_predictions(self, predictions, img_ids=None, time=None):

        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info(f"Preparing results for {self.dataset_name} format ...")
        mot_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        category_id=[]
        ## unmap the category ids for MOT
        #if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
        #    dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        #    all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
        #    num_classes = len(all_contiguous_ids)
        #    assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1
#
        #    reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        #    for result in mot_results:
        #        category_id = result["category_id"]
        #        
        #        assert category_id < num_classes, (
        #            f"A prediction has class={category_id}, "
        #            f"but the dataset only has {num_classes} classes and "
        #            f"predicted class id should be in [0, {num_classes - 1}]."
        #        )
        #        result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            if time:
                file_path = os.path.join(self._output_dir, self.dataset_name + f"_{time}_instances_results.json")
            else:
                file_path = os.path.join(self._output_dir, self.dataset_name + f"_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                self.results = mot_results
                self.results = pd.DataFrame(mot_results)
                self.results.bbox = [np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]], dtype=int) \
                                            for box in self.results.bbox]
                
                f.write(json.dumps(mot_results))
                f.flush()
                return self.mAP()

    def mAP(self):
        THRESH = 0.5
        boxes1 = self.results
        boxes1 = boxes1[boxes1.score>THRESH]
        metric = MeanAveragePrecision()
        for id in np.unique(boxes1.image_id):
            person_boxes = boxes1[np.logical_and(boxes1.category_id==0, boxes1.image_id==id)]
            #person_boxes = boxes1[boxes1.image_id==id]
            preds = [
            dict(
                boxes=torch.tensor(np.array(person_boxes.bbox.values.tolist())),
                scores=torch.tensor(person_boxes.score.values),
                labels=(torch.tensor(person_boxes.category_id.values)+1),
            )
            ]
            target = [
            dict(
                boxes=torch.tensor(self.gt[self.gt.image_id==id].iloc[:,2:6].values),
                labels=torch.tensor(self.gt[self.gt.image_id==id].iloc[:,7].values),
            )
            ]
            #target = [
            #dict(
            #    boxes=torch.tensor(self.gt[np.logical_and(self.gt[7]==1,self.gt.image_id==id)].iloc[:,2:6].values),
            #    labels=torch.tensor(self.gt[np.logical_and(self.gt[7]==1,self.gt.image_id==id)].iloc[:,7].values),
            #)
            #]
            
            metric.update(preds, target)
        
        df = pd.DataFrame()
        df[0] = metric.compute()
        return df[0].T
        

class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to DiffusionDet. """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.as_video = True if cfg.AS_VIDEO=='yes' else False
        self.time = cfg.TIME
        
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        

        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ########## EMA ############
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR
        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        #logger = logging.getLogger(__name__)
        #logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif 'coco' in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return MOTEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if 'MOT17' in cfg.DATASETS.TRAIN[0]:
            if not exists('datasets/MOT17'):
                raise ValueError('datasets/MOT17 directory does not exist')

            imgs = glob.glob('datasets/MOT17/train/*-DPM/img1/*')

            instances = glob.glob('datasets/MOT17/train/*-DPM/gt/*')

            dataset_dict=[]
            for vid in instances:
                config = configparser.ConfigParser()
                

                vid_id = vid.split('MOT17-')[-1].split('-')[0]
                config.read(f'datasets/MOT17/train/MOT17-{vid_id}-DPM/seqinfo.ini')
                gt = pd.read_csv(vid, header=None)
                img_dic={}
                train_ = int(np.unique(gt[0]).shape[0]*float(cfg.TRAIN_SIZE))
                for frame_id in np.sort(np.unique(gt[0]))[:train_]:
                    img_dic['file_name'] = np.array(imgs)[['MOT17-'+vid_id in i and str(frame_id).zfill(6) in i for i in imgs]][0]
                    img_dic['image_id'] = vid_id + '_' + str(frame_id)
                    annotations=[]
                    
                    for box in gt[gt[0]==frame_id].values:
                        ann = {}
                        box[4] = box[4] + box[2]
                        box[5] = box[5] + box[3]

                        box[4] = box[4].clip(0,int(config['Sequence']['imWidth']))
                        box[5] = box[5].clip(0,int(config['Sequence']['imHeight']))

                        ann['bbox'] = [box[2], box[3], box[4], box[5]]

                        ann['category_id'] = int(box[7])-1
                        ann['bbox_mode'] = BoxMode.XYXY_ABS
                        ann['iscrowd'] =  0
                        visibility = box[8]
                        if visibility>0.2 and ann['category_id']==0:
                            annotations.append(ann)
                    img_dic['annotations'] = annotations
                    dataset_dict.append(img_dic)

            
            dataset = DatasetFromList(dataset_dict, copy=False)
            mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
            dataset = MapDataset(dataset, mapper)
            sampler = TrainingSampler(len(dataset))
            dataloaders = torchdata.DataLoader(dataset,
                                                    batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                                    sampler=sampler,
                                                    drop_last=False,
                                                    collate_fn=trivial_batch_collator,
                                                )  
                                 
            return dataloaders
        else:
            mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if 'MOT' in dataset_name:

            if not exists('datasets/MOT17'):
                raise ValueError('datasets/MOT17 directory does not exist')

            imgs = glob.glob('datasets/MOT17/train/*-DPM/img1/*')
            dataset_dict = {}

            ## Building a datasetdict containing the information of of videos
            for img in imgs:
                try:
                    dataset_dict[img.split('MOT17-')[1][:2]].append({'image_id': int(img.split('.')[0].split('/')[-1]), 'file_name':img}) 
                except:
                    dataset_dict[img.split('MOT17-')[1][:2]] = [{'image_id': int(img.split('.')[0].split('/')[-1]), 'file_name':img}] 

            # Each video will have its own dataloader
            dataloaders = {}
            for vid in dataset_dict:
                train_ = int(len(dataset_dict[vid])*float(cfg.TRAIN_SIZE))
                dataset_dict[vid] = sorted(dataset_dict[vid], key=lambda x: x['image_id'])[train_:]
                dataset = DatasetFromList(dataset_dict[vid], copy=False)
                mapper = DatasetMapper(cfg, False)
                dataset = MapDataset(dataset, mapper)
                sampler = InferenceSampler(len(dataset))
                dataloaders[vid] = torchdata.DataLoader(dataset,
                                                        batch_size=1,
                                                        sampler=sampler,
                                                        drop_last=False,
                                                        collate_fn=trivial_batch_collator,
                                                    )
            return dataloaders
        else:
            return {'1':super().build_test_loader(cfg, dataset_name)}

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        The test function is an edited version of the DefaulTrainer test
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loaders = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            df_t = pd.DataFrame()
            
            # Going through all dataloaders
            for data_loader in data_loaders:
                if evaluators is not None:
                    evaluator = evaluators[idx]
                    
                print(f'---------------------------------{dataset_name}-{data_loader} Evaluation---------------------------------')
                if not exists(cfg.OUTPUT_DIR + '/inference'):
                    os.mkdir(cfg.OUTPUT_DIR + '/inference')
                if not exists(cfg.OUTPUT_DIR + '/inference/mAP'):
                    os.mkdir(cfg.OUTPUT_DIR + '/inference/mAP')
                # When at least one time is passed as argument
                if len(cfg.TIME)>0:
                    df = pd.DataFrame()
                    for i, t in enumerate(cfg.TIME):
                        if 'MOT' in dataset_name and cfg.AS_VIDEO=='yes':
                            dataset_name1 = dataset_name + '_' + data_loader + '_' + t[0]
                        elif 'MOT' in dataset_name: 
                            dataset_name1 = dataset_name + '_' + data_loader

                        evaluator = cls.build_evaluator(cfg, dataset_name1)
                        if t[0] == 'fbf': #fbf stands for frame by frame independentaly
                            cfg.T = None
                        else:
                            cfg.T = [t]

                        results_i = cls.inference_on_dataset(cfg, model, data_loaders[data_loader], evaluator)
                        
                        # Save the mAP metric in dataframe
                        df[t[0]] = results_i[0]

                    df = df.applymap(tensor_to_elem)
                    print(tabulate(df, headers='keys', tablefmt='psql'))
                
                    json_f = df.to_json()
                    file_path = os.path.join(cfg.OUTPUT_DIR + '/inference/mAP', dataset_name + '_' + data_loader + ".json")
                    
                    with PathManager.open(file_path, "w") as f:
                        f.write(json.dumps(json_f))
                        f.flush()
                        
                    results[data_loader] = df
                    if df_t.shape[0] == 0:
                        df_t = df
                    else:
                        for t in cfg.TIME:
                            df_t[t] = df_t[t] + df[t] 
                else:
                    if 'MOT' in dataset_name:
                        dataset_name1 = dataset_name + '_' + data_loader
                    else:
                        dataset_name1 = dataset_name

                    evaluator = cls.build_evaluator(cfg, dataset_name1)
                    results_i = cls.inference_on_dataset(cfg, model, data_loaders[data_loader], evaluator)
                    results[data_loader] = results_i
                    df_t = pd.DataFrame()
                    df_t[0] = results_i[0]

                    df_t = df_t.applymap(tensor_to_elem)
                    print(tabulate(df_t, headers='keys', tablefmt='psql'))
                
                json_f = df_t.to_json()
                
                file_path = os.path.join(cfg.OUTPUT_DIR + '/inference/mAP', dataset_name + ".json")
                
                with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps(json_f))
                    f.flush()

            if comm.is_main_process():
                assert isinstance(
                    results_i[1], dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i[1]
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))

        if len(results) == 1:
            results = list(results.values())[0]

        return results

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod 
    def inference_on_dataset(
    cls, cfg, model, data_loader, evaluator
):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.__call__` accurately.
        The model will be used in eval mode.

        Args:
            model (callable): a callable which takes an object from
                `data_loader` and returns some outputs.

                If it's an nn.Module, it will be temporarily set to `eval` mode.
                If you wish to evaluate a model in `training` mode instead, you can
                wrap the given model and override its behavior of `.eval()` and `.train()`.
            data_loader: an iterable object with a length.
                The elements it generates will be the inputs to the model.
            evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
                but don't want to do any evaluation.

        Returns:
            The return value of `evaluator.evaluate()`
        """
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} batches".format(len(data_loader)))

        total = len(data_loader)  # inference data loader must have a fixed length
        if evaluator is None:
            # create a no-op evaluator
            evaluator = DatasetEvaluators([])
        if isinstance(evaluator, abc.MutableSequence):
            evaluator = DatasetEvaluators(evaluator)
        evaluator.reset()
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            previous=None
            try:
                print(f'---------------------------------time={cfg.T}---------------------------------')
            except: pass
            for idx, inputs in tqdm(enumerate(data_loader)):
                if cfg.AS_VIDEO=='yes':
                    outputs, previous = model(inputs, previous, cfg.T)
                else:
                    outputs, _ = model(inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                evaluator.process(inputs, outputs)
        if cfg.T:
            results = evaluator.evaluate(cfg.T)
        else:
            results = evaluator.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return None

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

def tensor_to_elem(x):
    '''
    Change element of a dataframe 
    '''
    return x.item()

def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TEST = (args.dataset+'_val',)
    cfg.DATASETS.TRAIN = (args.dataset+'_train',)
    cfg.AS_VIDEO = args.as_video
    cfg.TIME = args.time
    cfg.T = None
    cfg.SELF_CONDITION = args.self_condition
    cfg.TRAIN_SIZE = args.train_size

    #cfg.freeze()
    #default_setup(cfg, args)
    
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        
        kwargs = may_get_ema_checkpointer(cfg, model)
        
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                              resume=args.resume)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS, 
                                                                                            resume=args.resume)
        
            
        
        res = Trainer.ema_test(cfg, model)

        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset", default='coco_2017', help="coco_2017, MOT17")
    parser.add_argument("--as-video", default='no', help="yes to take previous frame \
                                                            as intialisation in diffusiondet")
    parser.add_argument("--time", default=[], action='append', nargs='+', help="if as-video is yes, from which \
                                                    time you want to apply the diffusion process")
    parser.add_argument("--self-condition", default=False, help="Take the previous frame as condition if True") 
    parser.add_argument("--train-size", default=0.8)                                                
    
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
