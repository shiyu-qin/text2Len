import torch
import os
from options.TrainOptions import TrainLenEstOptions
from os.path import join as pjoin
import numpy as np
from utils.word_vectorizer import WordVectorizer, POS_enumerator
import utils.paramUtil as paramUtil
from networks.modules import MotionLenEstimator_1,MotionLenEstimator_2
from data.datasets import *
from torch.utils.data import DataLoader,random_split
from networks.trainers import *
if __name__=='__main__':
    parser = TrainLenEstOptions()
    opt = parser.parse()
    opt.device = torch.device('cuda:0')
    torch.autograd.set_detect_anomaly(True)
    # if opt.gpu_id!=-1:
    #     torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log', opt.dataset, opt.name)
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'eval')

    os.makedirs(opt.model_dir,exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.log_dir,exist_ok=True)

    if opt.dataset == 'kit':
        opt.data_root = '../xinglin-data/dataset/kit'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        if opt.bert == 'true':
            opt.text_dir = pjoin(opt.data_root, 'kit_labels')
            dim_word = 768
        else:
            opt.text_dir = pjoin(opt.data_root, 'texts')
            w_vectorizer = WordVectorizer('./glove', 'our_vab')
            dim_word = 300
            dim_pos_ohot = len(POS_enumerator)
            opt.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain
            num_classes = 200 // opt.unit_length
        opt.joints_num = 21
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    if opt.bert == 'true':
        print("opt.bert == true")
        estimator = MotionLenEstimator_2(dim_word, 512,200 // opt.unit_length)
        trainer = LengthEstTrainer_2(opt,estimator)
        train_dataset = LengthEstDataset(opt, train_split_file)
        print("train_dataset over")
        val_dataset = LengthEstDataset(opt, val_split_file)
        print("val_dataset over")
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, collate_fn=collate_fn_2,pin_memory=True)
        print("train_loader over")                      
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, collate_fn=collate_fn_2,pin_memory=True)
        print("val_loader over")                      
    else:
        estimator = MotionLenEstimator_1(dim_word, dim_pos_ohot, 512, num_classes)
        trainer = LengthEstTrainer_1(opt,estimator)
        train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, w_vectorizer)
        val_dataset = Text2MotionDataset(opt, mean, std, val_split_file, w_vectorizer)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, collate_fn=collate_fn_1, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, collate_fn=collate_fn_1, pin_memory=True)
    trainer.train(train_loader, val_loader)
    
    
