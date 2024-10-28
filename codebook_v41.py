
import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image

from utils.datasets import CodeBookDataset
from utils.modules import PositionalEncoding, Net,Net_multicls
from utils.plot_save import plot_save

import numpy as np
from tqdm import tqdm
import time
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(coordinate_path, codebook_path, exp_name, optim_type, lr,
         per_train, PoE, act_type, EPOCHS, batch_size,dim_hiddens, pre_random, summary_every,
         out_activation, vary_label, step_size, to11
         ):
# def main(coordinate_path, codebook_path, exp_name, optim_type="sgd", lr=1e-4,
#          per_train=0.8, PoE=None, act_type="Sin", EPOCHS=100, batch_size=0,
#          dim_hiddens = [], pre_random = False,summary_every = 100,
#          out_activation = "NoAF",vary_label = None,step_size=500
#
#          ):
    """
    Args:
        coordinate_path (str):
        codebook_path (str):
        exp_name (str):
        opt:
        lr:
        per_train:
        PoE:
        act_type:
        EPOCHS:
        batch_size:
    Returns:

    """

    to_cuda = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 该次实验名称
    
    # 数据集
    # np.load数据集
    codebook = np.load(codebook_enc_path)
    coordinate = np.load(coordinate_path)
    print(coordinate.shape)

    print("\ncoordinate.shape0", coordinate.shape)
    print("codebook.shape0", codebook.shape)

    dim_coor = coordinate.shape[1]
    dim_codebook = codebook.shape[1]  # 码本的维度
    if len(codebook.shape) == 3:
        dim_codebook = dim_codebook * codebook.shape[2]  # 码本的维度
    
    # 划分数据集-训练集
    if pre_random:
        row_indices = np.load(os.path.join(os.path.dirname(codebook_path), 'row_indices.npy'))
    else:
        row_indices = np.random.permutation(coordinate.shape[0])
    
    if per_train < 1:
        print(f"num of data sample : {len(codebook)} training percent {per_train * 100}%")
        len_train = int(len(coordinate) * per_train)
    else:
        len_train = per_train
        print(f"num of data sample : {len(codebook)} training percent {len_train/len(codebook)}%")
    train_index = row_indices[0:len_train]
    test_index = row_indices[len_train:]
    train_coor, train_code = coordinate[train_index, :], codebook[train_index]
    test_coor, test_code = coordinate[test_index, :], codebook[test_index]

    if PoE is not None:
        PE = PositionalEncoding(PoE[0], PoE[1])
        print(f"Positional encoding : b={PoE[0]}, l={PoE[1]}")
        dim_input = PoE[1] * 2 * dim_coor
    else:
        PE = None
        print(f"Positional encoding : None")
        dim_input = dim_coor
    print("dim_input:", dim_input)
    
    train_datasets = CodeBookDataset(train_coor, train_code, PE, vary_label=vary_label, to_cuda=to_cuda)
    test_datasets = CodeBookDataset(test_coor, test_code, PE, vary_label=vary_label, to_cuda=False)
    
    if batch_size == 0:
        batch_size = len(train_datasets)
    
    train_loader = DataLoader(train_datasets, batch_size=batch_size, pin_memory=False, num_workers=0,
                              shuffle=True, drop_last=False)
    test_loader = DataLoader(test_datasets, batch_size=len(test_datasets), pin_memory=True, num_workers=0)

    # train_coordinate, train_codebook = next(iter(train_loader))
    # train_coordinate, train_codebook = train_coordinate.to(DEVICE), train_codebook.to(DEVICE)

    test_coordinate, test_codebook = next(iter(test_loader))
    test_coordinate, test_codebook = test_coordinate.float().to(DEVICE), test_codebook.long().to(DEVICE)
    test_codebook = test_codebook.long()
    all_unit = test_codebook.shape[0] * test_codebook.shape[1]

    codebook_net = Net_multicls(dim_in=dim_input, dim_hiddens=dim_hiddens, dim_out=dim_codebook,
                       first_omega_0=30., hidden_omega_0=30., act_type=act_type, out_activation=out_activation).to(DEVICE)
    print(codebook_net)
    print(f"batch_size: {batch_size}")
    
    if optim_type == "sgd":
        print("USE SGD as optimzer")
        optim = torch.optim.SGD(params=codebook_net.parameters(), lr=lr, momentum=0.9)  # 例化对象
    else:
        print("USE Adam as optimzer")
        optim = torch.optim.Adam(lr=lr, params=codebook_net.parameters())
    print(f"number of train: {len(train_datasets)}\nnumber of test: {len(test_datasets)}")
    
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=0.5)
    
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.MultiLabelMarginLoss()
    # loss_fn = nn.MSELoss(reduction='mean')
    min_loss_val = 100000
    min_epoch = 10
    data_dict = {"losses_train":[], "losses_test":[], "acc_test": [], "totalacc_test" : []}
    codebook_net.cuda()
    bar = tqdm(range(1, EPOCHS + 1))
    
    best_acc = 0
    
    for epoch in bar:
        loss_epoch = []
        codebook_net.train()
        for train_coordinate, train_codebook in train_loader:
            # print(train_codebook.shape)
            train_coordinate, train_codebook = train_coordinate.to(DEVICE), train_codebook.long().to(DEVICE)
            # train_coordinate, train_codebook = train_coordinate.to(DEVICE), train_codebook.to(DEVICE)
            optim.zero_grad()
            
            pre_codebook = codebook_net(train_coordinate)
            # print(pre_codebook)
            loss = loss_fn(pre_codebook, train_codebook)
            loss.backward()
            optim.step()
            loss_epoch.append(loss.cpu().detach().numpy().item())
        scheduler.step()
        train_loss_v = np.mean(loss_epoch)
        data_dict["losses_train"].append(train_loss_v)
        codebook_net.eval()
        with torch.no_grad():
            test_pre = codebook_net(test_coordinate)
            test_loss = loss_fn(test_pre, test_codebook)
            test_loss_val = test_loss.cpu().detach().numpy().item()
            if epoch > min_epoch and test_loss_val <= min_loss_val:
                min_loss_val = test_loss_val
                best_model = copy.deepcopy(codebook_net)
            data_dict["losses_test"].append(test_loss_val)

            test_pre = test_pre.argmax(dim = 1)

            equals = test_pre.eq(test_codebook)  # 相等为True
            
            correct = equals.sum().item()  # 完全正确样本个数
            
            acc_the = correct / all_unit  # 除去总单元数得到单元正确率
            if acc_the > best_acc:
                best_acc = acc_the
                best_model = copy.deepcopy(codebook_net)
            
            data_dict["acc_test"].append(acc_the)  # 平均每个码本的正确率

            per_right = equals.sum(1)  # 每个样本的正确单元数
            # 如果单个样本正确单元数等于单元数
            per_code = (per_right == dim_codebook).sum().item()  # 预测完全正确样本数

            data_dict["totalacc_test"].append(per_code / len(test_coor))  # 完全预测正确准确率
            
            bar.set_postfix(train_loss = "{:.5f}".format(train_loss_v), test_acc = "{:.2f}%".format(100*acc_the))
            
            if epoch % summary_every == 0:
                print('\nTrain Epoch: {} \tLoss: {:.6f}, learn rate : {:.6f}'.format(epoch, loss.item(), optim.state_dict()['param_groups'][0]['lr']))
                print(
                    'Test set: Average loss: {:.4f}, Accuracy: unit: {}/{} ({:.0f}%) -- sample {}/{} ({:.0f}%)\n'.format(
                        test_loss, correct, all_unit, 100 * acc_the,
                        per_code, len(test_coor), 100. * per_code / len(test_coor)))
    
    infer = best_model(test_coordinate)
    
    exp_setting = "{}-{}-{}-B{}-hL{}-lr{}-E{}-p{}_".format(act_type, PoE, optim_type, batch_size, dim_hiddens[-1], lr, EPOCHS, per_train)
    cnt = 0
    exp_nu='exp_'
    while os.path.exists(os.path.join('run', exp_name, exp_nu + str(cnt).zfill(2))):
        cnt = cnt + 1
    exp_dir = os.path.join('run', exp_name, exp_nu + str(cnt).zfill(2))
    os.makedirs(exp_dir)
    
    # torch.save(best_model, os.path.join(exp_dir, "best.pt"))
    
    np.save(os.path.join(exp_dir, 'results_data-%.4f.npy' % max(data_dict["acc_test"])), data_dict)    #在当前目录生成test.npy文件
    np.save(os.path.join(exp_dir, 'equals.npy'),equals.cpu().numpy())
    
    np.save(os.path.join(exp_dir, 'test_infer_enc.npy'), infer.cpu().detach().numpy())
    
    np.save(os.path.join(exp_dir, 'train_index.npy'), train_index)
    np.save(os.path.join(exp_dir, 'test_index.npy'), test_index)

    np.save(os.path.join(exp_dir, 'row_indices.npy'), row_indices)

    exp_title = f"B{batch_size} {act_type} PE:{PoE} opt:{optim_type}"

    plot_save(data_dict, exp_title, os.path.join(exp_dir, 'loss_acc.pdf'))

    data_dict["losses_train"] = np.log10(data_dict["losses_train"])
    data_dict["losses_test"] = np.log10(data_dict["losses_test"])

    plot_save(data_dict, exp_title, os.path.join(exp_dir, 'log10 loss_acc.pdf'))

    results_file_path = os.path.join(exp_dir, 'config.txt')
    print("save to ", results_file_path)
    with open(os.path.join(exp_dir, exp_setting+'.txt'), 'w') as f:
        f.write(exp_setting)
    
    f = open(results_file_path, 'w')

    f.write("#" * 30 + " Training Info " + "#" * 30)
    f.write(f"\n - experiment name : {exp_name}")
    f.write(f"\n - datasets root : {coordinate_path}")
    f.write(f"\n - traing EPOCHS : {EPOCHS}")
    f.write(f"\n - batch size : {batch_size}")
    f.write(f"\n - activate function : {act_type}")
    f.write(f"\n - optimizer : {optim_type}")
    f.write(f"\n - learn rate : {lr}")
    f.write(f"\n - train time : {str(bar).split('[')[-1][:-1]}")
    f.write(f"\n - loss funcition : MSE")
    f.write(f"\n - Net : \n{str(codebook_net)}\n")
    f.write("#" * 30 + " Training Info " + "#" * 30)
    f.close()
    return max(data_dict["totalacc_test"]), max(data_dict["acc_test"]), min(data_dict["losses_train"]), min(data_dict["losses_test"])


# def exp_struct(data_dir, exp_name, enc = 0):
#     if enc == 0:
#         codebook_enc_path = os.path.join(data_dir, "codebookEnc_list0.npy") # 旧编码
#     else:
#         codebook_enc_path = os.path.join(data_dir, "codebookEnc_list.npy")
#     codebook_path = os.path.join(data_dir, "codebook_list.npy")
#     coordinate_path = os.path.join(data_dir, "coordinate_list.npy")
#
#     EPOCHS = 1000
#     # exp_dict = {"ReLU": {"PoE": None, "act_type": "ReLU", "lr":1e-4, "optim_type":'adam'},
#     #             "ReLU+PE": {"PoE": (1.25, 20), "act_type": "ReLU", "lr":1e-4, "optim_type":'adam'},
#     #             "Sin": {"PoE": None, "act_type": "Sin", "lr":1e-3, "optim_type":'sgd'},
#     #             "Sin+PE": {"PoE": (1.25, 20), "act_type": "Sin", "lr":1e-3, "optim_type":'sgd'},
#     #             "Tanh": {"PoE": None, "act_type": "Tanh", "lr":1e-3, "optim_type":'adam'},
#     #             "Tanh+PE": {"PoE": (1.25, 20), "act_type": "Tanh", "lr":1e-3, "optim_type":'adam'}
#     #             }
#     # result_dict = {"ReLU": [], "ReLU+PE": [], "Sin": [], "Sin+PE": [], "Tanh": [], "Tanh+PE": []}
#     exp_dict = {"LeakyReLU": {"PoE": None, "act_type": "LeakyReLU", "lr":5e-2, "optim_type":'adam'},
#                 "LeakyReLU+PE": {"PoE": (1.25, 120), "act_type": "ReLU", "lr":5e-2, "optim_type":'adam'},
#                 "GELU": {"PoE": None, "act_type": "Tanh", "lr":1e-3, "optim_type":'adam'},
#                 "GELU+PE": {"PoE": (1.25, 120), "act_type": "Tanh", "lr":1e-3, "optim_type":'adam'},
#                 }
#     result_dict = {"ReLeakyReLULU": [], "LeakyReLU+PE": [], "GELU": [], "GELU+PE": []}
#
#     dim_hiddens = [128,128,128,128]
#     train_per = 0.8
#     for key in exp_dict:
#         # lr = exp_dict[key]["lr"]
#         acc = main(coordinate_path, codebook_enc_path, exp_name, per_train=train_per,
#                PoE=exp_dict[key]["PoE"], act_type=exp_dict[key]["act_type"], EPOCHS=EPOCHS, batch_size=128,
#                dim_hiddens=dim_hiddens, optim_type=exp_dict[key]["optim_type"], lr=exp_dict[key]["lr"], pre_random=True, outermost_linear=True)
#         result_dict[key].append(acc)
#     print(result_dict)
#     np.save(os.path.join("run", exp_name, 'result_all_acc.npy'), result_dict)

#
if __name__ == '__main__':
    data_dir = r'data\measured0421_nozero'
    #data_dir = r'data\2023-04-23'
    codebook_enc_path = os.path.join(data_dir, "codebookEnc_list.npy")
    codebook_path = os.path.join(data_dir, "codebook_list.npy")
    coordinate_path = os.path.join(data_dir, "coordinate_list.npy")

    train_per = 0.9
    lr = 1e-1
    optim_type = "sgd"
    #act_type = "Softsign"
    act_type = "ReLU"
    out_activation = "NoAF"
    # out_activation = "Softsign"
    epochs = 1000
    dim_hiddens = [128,128,128,128]
    exp_name = "times2_test"

    # [1.5,120] None
    vary_label = None
    acc = main(coordinate_path, codebook_path, exp_name, per_train=train_per,
               PoE=[1.5,60], act_type=act_type, EPOCHS=epochs, batch_size=256,
               dim_hiddens=dim_hiddens, optim_type=optim_type, lr=lr, pre_random=True,
               summary_every=100, out_activation=out_activation, vary_label=vary_label, step_size=500
               , to11=False)
    # acc = main(coordinate_path, codebook_enc_path, exp_name, per_train=train_per,
    #         PoE=[1.5,120], act_type=act_type, EPOCHS=epochs, batch_size=64,
    #         dim_hiddens=dim_hiddens, optim_type=optim_type, lr=lr, pre_random=False,
    #         out_activation=out_activation,vary_label=vary_label,step_size=500)
    print(acc)


# def exp_lr():
#     # 测试不同的学习率
#     data_dir = r'data\2023-04-23'
#     #data_dir = r'data\measured0421_nozero'
#     codebook_enc_path = os.path.join(data_dir, "codebookEnc_list.npy")
#     codebook_path = os.path.join(data_dir, "codebook_list.npy")
#     coordinate_path = os.path.join(data_dir, "coordinate_list.npy")
#
#     EPOCHS = 100
#
#     lr = 1e-1
#     optim_type = "adam"
#     act_type = "ReLU"
#     out_activation = "NoAF"
#     epochs = 1000
#     # dim_hiddens = [360,360,256,256,128]
#     dim_hiddens = [128,128,128,128]
#
#     # exp_name = "TrainIN-SIM-PE1-Tanh-ENC1"
#     vary_label = [0,1]
#     PoE = [1.5,40]
#     #PoE = None
#     exp_name = f"TrainIN-Measured-PE{PoE}-{act_type}-ENC1"
#
#     dim_hiddens = [256,256,256,256,256]
#     train_per = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     result_dict = {}
#     for per_i in train_per:
#         acc = main(coordinate_path, codebook_path, exp_name, per_train=per_i,
#                 PoE=PoE, act_type=act_type, EPOCHS=EPOCHS, batch_size=128,
#                 dim_hiddens=dim_hiddens, optim_type=optim_type, lr=lr, pre_random=True,
#                 out_activation=out_activation,vary_label=vary_label,step_size=400)
#         result_dict[per_i] = acc[1]
#     print(result_dict)
#     np.save(os.path.join("run", exp_name, 'lrs_acc.npy'), result_dict)
#
# exp_lr()
#
#
