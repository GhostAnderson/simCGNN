from model import *
from utils import Data, split_validation
import torch
import random
import pickle
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--seed', type=int, default=3407, help='random seed')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)

test_data = pickle.load(open('../datasets/yoochoose1_64/test.txt', 'rb'))
test_data = Data(test_data)
opt.len_max = 145
n_node = 37484

model = trans_to_cuda(SessionGraph(opt, n_node))
model.load_state_dict(torch.load('./final_yc/epoch_5.pth'))
print('---model loaded---')

model.eval()
slices = test_data.generate_batch(model.batch_size)
res = []
last_items = np.array(test_data.last_items)
for i in tqdm.tqdm(slices):
    targets, scores, global_session = forward(model, i, test_data)
    sub_scores = scores.topk(20)[1]

    targets = (targets-1).tolist()
    l_i = last_items[i].tolist()
    sub_scores = trans_to_cpu(sub_scores).detach().numpy().reshape(-1,20).tolist()

    res += list(zip(l_i, targets, sub_scores))

np.save('res.yc.npy', res)