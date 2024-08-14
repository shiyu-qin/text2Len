import argparse
class TrainLenEstOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--dataset',type=str,default='kit',help='name of dataset')
        self.parser.add_argument("--gpu_id", type=int, default=0,help='GPU id')
        self.parser.add_argument('--max_epoch',type=int,default=300,help='训练次数')
        self.parser.add_argument('--checkpoints_dir', type=str, default='../xinglin-data/checkpoints', help='models are saved here')
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument('--bert', type=str, default="true", help='is use bert')
        self.parser.add_argument('--batch_size', type=int, default=4, help='每一次训练的样本数')
        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')
        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')
        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')
        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")
    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        args = vars(self.opt)
        return self.opt
