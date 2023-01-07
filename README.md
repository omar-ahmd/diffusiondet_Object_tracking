#   DiffsuionDet for object tracking
This code is based on the implementation of [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet), [BoT-SORT](https://github.com/NirAharon/BoT-SORT#bot-sort)
The main contribution of this work is to use the previous frame as an intialization in the diffusion process to imporve its preformance on object tracking, and to prepare the enviroment to train the diffusiondet on MOT dataset

# 1.Data Preparation
Download [MOT17](https://motchallenge.net/data/MOT17/) from the [official website](https://motchallenge.net/). And put them in the following structure:

```
<dataets_dir>
      │
      ├── MOT17
             ├── train
             └── test    
      
```   
                    
# 2.Training DiffsuionDet
Single GPU training
```
cd <prb_dir>
$ python train_net_MOT.py --num-gpus 1 --config-file <coco config file> --dataset MOT17 --resume  MODEL.DiffusionDet.SAMPLE_STEP 1 MODEL.DEVICE cuda MODEL.WEIGHTS <coco weights>  SOLVER.IMS_PER_BATCH 4 MODEL.DiffusionDet.NUM_CLASSES 1
```

# 3.Inference
```
cd <prb_dir>
$ python train_net_MOT.py --num-gpus 1 --config-file <coco config file> --dataset MOT17  --as-video yes --time 100 --eval-only MODEL.DiffusionDet.SAMPLE_STEP 1 MODEL.DEVICE cuda MODEL.WEIGHTS <MOT weights> MODEL.DiffusionDet.NUM_CLASSES 1
```
as-video: use or not the previous frame as prior\
time: the time from which you want to start the diffusion process with the use of the previous frame as prior


# 4.Tracking
```
cd <prb_dir>
$ python BoT-SORT/tools/track_diffdet.py ./datasets/MOT17 --default-parameters --eval val --config-diffdet-file <coco config file>  --weights-file <finetuned MOT weights> --time 100
```
time: the time from which you want to start your diffusion process with the use of the previous frame as prior


# 5.Acknowledgement
A large part of the codes are borrowed from [DiffsuionDet](https://github.com/ShoufaChen/DiffusionDet), [BoT-SORT](https://github.com/NirAharon/BoT-SORT#bot-sort), thanks for their excellent work!