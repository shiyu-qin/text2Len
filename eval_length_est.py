import os

from os.path import join as pjoin

from options.TrainOptions import TrainLenEstOptions

from networks.modules import *
from networks.trainers import LengthEstTrainer_1
from data.datasets import Text2MotionDataset,collate_fn_1,collate_fn_2,RawTextDataset,LengthEstDataset
# from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator
import torch.nn as nn
import codecs as cs
import utils.paramUtil as paramUtil
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
if __name__ == '__main__':
    parser = TrainLenEstOptions()
    opt = parser.parse()
    # opt.is_train = False

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')

    if opt.dataset == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)
        num_classes = 200 // opt.unit_length

        mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        std = np.load(pjoin(opt.data_root, 'Std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'val.txt')
    elif opt.dataset == 'kit':
        opt.data_root = '../xinglin-data/dataset/kit'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        if opt.bert == 'true':
            opt.text_dir = pjoin(opt.data_root, 'texts2/texts2')
            dim_word = 3072
        else:
            opt.text_dir = pjoin(opt.data_root, 'texts')
            w_vectorizer = WordVectorizer('./glove', 'our_vab')
            dim_word = 300
            dim_pos_ohot = len(POS_enumerator)
            opt.max_motion_length = 196
            num_classes = 200 // opt.unit_length
            kinematic_chain = paramUtil.kit_kinematic_chain
        opt.joints_num = 21
        dim_pose = 251
    else:
        raise KeyError('Dataset Does Not Exist')
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))
    split_file = pjoin(opt.data_root, 'test.txt')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    if opt.bert != 'true':
        estimator = MotionLenEstimator_1(dim_word, dim_pos_ohot, 512, num_classes)
        dataset = Text2MotionDataset(opt, mean, std, split_file, w_vectorizer)
        loader = DataLoader(dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, collate_fn=collate_fn_1, pin_memory=True) 
    else:
        estimator = MotionLenEstimator_2(dim_word, 512,200 // opt.unit_length)
        dataset = LengthEstDataset(opt, split_file)
        loader = DataLoader(dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, collate_fn=collate_fn_2, pin_memory=True)
    # else:
    #     raise Exception('Estimator Mode is not Recognized!!!')

    # loader = DataLoader(dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,shuffle=True, collate_fn=collate_fn_1, pin_memory=True)
    # checkpoints = torch.load(pjoin(opt.model_dir, 'finest.tar'),weights_only=True)
    checkpoints = torch.load('/root/xinglin-data/checkpoints/kit/bertT3/model/finest.tar',weights_only=True)
    estimator.load_state_dict(checkpoints['estimator'])
    estimator.to(opt.device)
    
    # estimator.eval()

    softmax = nn.Softmax(dim=1)
    save_path = pjoin('eval_results', opt.dataset, opt.name, 'test')
    os.makedirs(save_path, exist_ok=True)
    mses = []
    array_data = []
    array_label = []
    r2s = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            # if i >0:
            #     break
            # data.to(opt.device)
            if opt.bert != 'true':
                word_emb, pos_ohot, caption, cap_lens, _, m_lens = data
            # print("opt.device",opt.device)
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()
                pred_dis = estimator(word_emb, pos_ohot, cap_lens)
            else:
                word_embeddings,_,real_len,m_lens = data
                word_emb = torch.squeeze(word_embeddings, dim=1).detach().to(opt.device).float()
                pred_dis = estimator(word_emb, real_len)
            # print("pred_dis",pred_dis.size(),type(pred_dis))
            pred_dis = softmax(pred_dis).cpu().numpy()
            # print("pred_dis",pred_dis.shape,type(pred_dis))
            predicted_classes = np.argmax(pred_dis,axis =1 )
            # print("predicted_classes",predicted_classes,type(predicted_classes))
            # print("m_lens",m_lens//opt.unit_length,type(m_lens))
            motion_lens = (m_lens//opt.unit_length).cpu().numpy()
            mse = np.mean((predicted_classes - motion_lens) ** 2)
            r2 = r2_score(predicted_classes, motion_lens)
            r2s.append(r2)
            array_data.append(predicted_classes)
            array_label.append(motion_lens)
            
            rmse = np.sqrt(mse)
            mses.append(rmse)
            # print(f"Iteration {i}, RMSE: {rmse}")
    print("均方误差",np.mean(rmse))
    # r2 = r2_score(array_data, array_label)
    print("r2",np.mean(r2s),type(r2))
    #     # 绘制均方误差
    # plt.plot(mses, marker='o')
    # plt.xlabel('Iteration')
    # plt.ylabel('Root Mean Squared Error (RMSE)')
    # plt.title('Root Mean Squared Error Over Iterations')
    # plt.grid(True)
    
    # # 保存图表到文件
    # plt.savefig(save_path+'/mse_over_iterations_1.png')  # 保存为 PNG 文件