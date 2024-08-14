import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from os.path import join as pjoin
import time
import numpy as np
from collections import OrderedDict
from utils.utils import print_current_loss_decomp
import matplotlib.pyplot as plt
class Logger(object):
  def __init__(self, log_dir):
    # self.writer = tf.summary.create_file_writer(log_dir)
    pass

  def scalar_summary(self, tag, value, step):
    #   with self.writer.as_default():
    #       tf.summary.scalar(tag, value, step=step)
    #       self.writer.flush()
    pass
class LengthEstTrainer_1(object):

    def __init__(self, args, estimator):
        self.opt = args
        self.estimator = estimator
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            self.logger = Logger(args.log_dir)
            self.mul_cls_criterion = torch.nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def train(self, train_dataloader, val_dataloader):
        self.estimator.to(self.device)

        self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        logs = OrderedDict({'loss': 0})
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.estimator.train()

                word_emb, pos_ohot, _, cap_lens, _, m_lens = batch_data
                word_emb = word_emb.detach().to(self.device).float()
                pos_ohot = pos_ohot.detach().to(self.device).float()

                pred_dis = self.estimator(word_emb, pos_ohot, cap_lens)

                self.zero_grad([self.opt_estimator])

                gt_labels = m_lens // self.opt.unit_length
                gt_labels = gt_labels.long().to(self.device)
                # print(gt_labels)
                # print(pred_dis)
                loss = self.mul_cls_criterion(pred_dis, gt_labels)

                loss.backward()

                self.clip_norm([self.estimator])
                self.step([self.opt_estimator])

                logs['loss'] += loss.item()

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.logger.scalar_summary('val_loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict({'loss': 0})
                    print_current_loss_decomp(start_time, it, total_iters, mean_loss, epoch, i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, it)

            print('Validation time:')

            val_loss = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    word_emb, pos_ohot, _, cap_lens, _, m_lens = batch_data
                    word_emb = word_emb.detach().to(self.device).float()
                    pos_ohot = pos_ohot.detach().to(self.device).float()

                    pred_dis = self.estimator(word_emb, pos_ohot, cap_lens)

                    gt_labels = m_lens // self.opt.unit_length
                    gt_labels = gt_labels.long().to(self.device)
                    loss = self.mul_cls_criterion(pred_dis, gt_labels)

                    val_loss += loss.item()

            val_loss = val_loss / (len(val_dataloader) + 1)
            print('Validation Loss: %.5f' % (val_loss))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss

'''估计文本生成的运动序列的长度'''
class LengthEstTrainer_2(object):
    def __init__(self,args,estimator):
        self.opt = args
        self.estimator = estimator
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.mul_cls_criterion = nn.CrossEntropyLoss()

    # ：加载之前保存的模型参数和优化器状态，恢复训练。
    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)
    #清零优化器的梯度。 
    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    # 剪裁梯度以防止梯度爆炸。
    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    #执行优化器的更新步骤。
    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()
    
    # 训练和验证
    def train(self, train_dataloader, val_dataloader):
        print("正在开启训练.....{bert is true}")
        self.estimator.to(self.device)
        print(torch.cuda.current_device())
        self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        # OrderedDict 是 Python 中的一个数据结构，它是 collections 模块中的一部分。
        # 它类似于普通的字典 dict，但是它记住了元素被添加的顺序，因此可以按照元素被添加的顺序进行迭代。
        logs = OrderedDict({'loss': 0})
        
        train_loss = []
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            runningloss = 0.0
            for i, batch_data in enumerate(train_dataloader):
                self.estimator.train()
                # batch_data.to(self.device)
                # word_embeddings, _, cap_lens, _, m_lens = batch_data
                word_embeddings,_,real_len,m_lens = batch_data
                # detach() 方法创建一个新的张量，其值与原始张量相同，但是不再与计算图相关联。
                # 这意味着该张量不会再跟踪其计算历史，不会有梯度信息。
                word_emb = torch.squeeze(word_embeddings, dim=1).detach().to(self.device).float()
                word_emb = word_emb.to(self.device)
                real_len = real_len.to(self.device)
                pred_dis = self.estimator(word_emb, real_len)#[9, 10, 11, 9]
                # print("pred_dis",pred_dis.size())
                self.zero_grad([self.opt_estimator])               
                # 这里，增量 1 相当于 4 个姿势帧，设置 Tmax = 50 相当于 200 个帧，
                # 对于 20 帧/秒的视频而言，相当于 10 秒钟。其训练目标由交叉熵损失定义
                gt_labels = m_lens // self.opt.unit_length
                gt_labels = gt_labels.long().to(self.device)
                loss = self.mul_cls_criterion(pred_dis, gt_labels)
                # print("loss",loss)
                loss.backward()
                # print("after loss")
                
                # self.clip_norm([self.estimator])
                self.step([self.opt_estimator])

                logs['loss'] += loss.item()
                runningloss += loss.item()

                it += 1
                if it % self.opt.log_every == 0:
                    # mean_loss = OrderedDict({'val_loss': val_loss})
                    # # self.logger.scalar_summary('val_loss', val_loss, it)

                    # for tag, value in logs.items():
                    #     self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                    #     # mean_loss[tag] = value / self.opt.log_every
                    # logs = OrderedDict({'loss': 0})
                    # print_current_loss_decomp(start_time, it, total_iters, train_loss, epoch, i)
                    # print("f'Epoch {epoch+1}/{self.opt.max_epoch}:{i}, Loss: {runningloss:.4f}'")
                    # print_current_loss_decomp(start_time, it, total_iters, train_loss/len(train_dataloader.dataset), epoch, i)
                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            
            runningloss = runningloss/(len(train_dataloader) + 1)
            train_loss.append(runningloss)
            print(f'Epoch {epoch+1}/{self.opt.max_epoch}, Loss: {runningloss:.4f}')
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            if runningloss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = runningloss
            epoch += 1
            # if epoch % self.opt.save_every_e == 0:
            #     self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, it)

            # print('Validation time:')

        # test_loss = 0
        # with torch.no_grad():
        #     for i, batch_data in enumerate(val_dataloader):
        #         # word_emb, pos_ohot, _, cap_lens, _, m_lens = batch_data
        #         word_embeddings,_,real_len,m_lens = batch_data
        #         word_emb = torch.squeeze(word_embeddings, dim=1).detach().to(self.device).float()
        #         # pos_ohot = pos_ohot.detach().to(self.device).float()
        #         word_emb = word_emb.to(self.device)
        #         real_len = real_len.to(self.device)
        #           # 
        #         pred_dis = self.estimator(word_emb, real_len)
        #           # 
        #         gt_labels = m_lens // self.opt.unit_length
        #         gt_labels = gt_labels.long().to(self.device)
        #         loss = self.mul_cls_criterion(pred_dis, gt_labels)
        #         test_loss += loss.item()
        # test_loss = test_loss / (len(val_dataloader) + 1)
        # print('Validation Loss: %.5f' % (val_loss))

        plt.plot(train_loss, marker='o')
        plt.title('Traing_loss of Kit')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('/root/text2len_v02/eval_results/line_plot.png')

            # if val_loss < min_val_loss:
            #     self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
            #     min_val_loss = val_loss