import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import new_method_dataset, time_dataset
from new_model import fra_sig_network, time_model
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, f1_score
from itertools import cycle
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import math
import matplotlib
matplotlib.use('TkAgg')


def load_data():
    label_train = []
    label_test = []
    with open('./data/vir/images/train.txt', 'r') as f:
        line_train = f.readlines()
    with open('./data/vir/images/test.txt', 'r') as f:
        line_test = f.readlines()
    for name in line_train:
        label_train.append(int(name.split('/')[-1][0]))
    for name in line_test:
        label_test.append(int(name.split('/')[-1][0]))
    return line_train, label_train, line_test, label_test


def train():
    print('*' * 5 + 'start training' + '*' * 5)
    class_num = [0, 1, 2, 3, 4, 5, 6]
    train_acc = []
    val_acc = []
    train_total_loss = []
    val_total_loss = []
    val_precision = []
    val_recall = []
    best_val_acc = 0
    best_train_acc = 0

    for epoch in range(num_epoch):
        print('-' * 10 + 'epoch ' + str(epoch + 1) + '/' + str(num_epoch) + '-' * 10)
        epoch_train_acc = 0
        label_pre = []
        label_true = []
        train_loss = 0
        val_loss = 0
        epoch_val_acc = 0

        net.train()
        net.mode = 'train'
        for i, data in enumerate(train_dataloader):
            batch_acc = 0
            img_sig, label = data['image_sig'], data['label']
            img_sig = img_sig.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            if torch.cuda.is_available():
                img_sig = Variable(img_sig.cuda(), requires_grad=False)
                label = Variable(label.cuda(), requires_grad=False)

            optimizer.zero_grad()
            res_pre = net(img_sig)
            loss = fun_loss(res_pre, label)
            train_loss += loss
            loss.backward()
            optimizer.step()

            # 计算准确率
            for j in range(res_pre.shape[0]):
                if torch.argmax(res_pre[j]) == torch.argmax(label[j]):
                    epoch_train_acc = epoch_train_acc + 1
                    batch_acc += 1
                # print('{}/{}:{}'.format((i + 1) * train_batch, len(train_fold), batch_acc / img_sig.shape[0]))

        # 保存模型
        if epoch_train_acc / len(train_dataset) >= best_train_acc:
            best_train_acc = epoch_train_acc / len(train_dataset)
            torch.save(net.state_dict(), './models_save/pu_time.pth')

        '''label_pre = label_binarize(label_pre, classes=[i for i in range(7)])
        label_true = label_binarize(label_true, classes=[i for i in range(7)])
        precision = precision_score(label_true, label_pre, average='macro')
        recall = recall_score(label_true, label_pre, average='macro')'''

        train_acc.append(epoch_train_acc / len(train_dataset))
        train_total_loss.append(train_loss / (len(train_dataset) / batch_size_train))
        print('train loss：{}'.format(train_loss / math.ceil((len(train_dataset) / batch_size_train))))
        print('train acc：{}'.format(epoch_train_acc / len(train_dataset)))
    print('training end，the best acc on trainset：{}, the best acc on valset: {}'.format(best_train_acc, best_val_acc))




if __name__ == '__main__':
    train_data, train_label, val_data, val_label = load_data()
    print('*' * 10)
    print('训练集数量：', len(train_label))
    print('测试集数量：', len(val_label))

    # 损失函数
    fun_loss = nn.CrossEntropyLoss()

    # 创建数据集
    batch_size_train = 32
    num_epoch = 50

    train_dataset = time_dataset(data=train_data, label=train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=10)


    # 初始化模型
    net = time_model(num_out=7, mode='train')
    '''save_model = torch.load('./models_save/con_vit_vir.pth')
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)'''
    if torch.cuda.is_available():
        net.cuda()

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)

    # 训练
    train()
