import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import new_method_dataset
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
    with open('./data/vir/images/train.txt') as f:
        line_train = f.readlines()
    with open('./data/vir/images/test.txt') as f:
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

    kf = KFold(n_splits=5, shuffle=True)
    k=0
    for train, val in kf.split(train_dataset):
        best_val_acc = 0
        best_train_acc = 0
        k += 1
        train_fold = torch.utils.data.dataset.Subset(train_dataset, train)
        val_fold = torch.utils.data.dataset.Subset(train_dataset, train)
        train_dataloader = DataLoader(train_fold, batch_size=train_batch, shuffle=True, num_workers=10)
        val_dataloader = DataLoader(val_fold, batch_size=train_batch, shuffle=True, num_workers=10)

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
                img_fra, img_sig, label = data['image_fra'], data['image_sig'], data['label']
                img_fra = img_fra.type(torch.FloatTensor)
                img_sig = img_sig.type(torch.FloatTensor)
                label = label.type(torch.FloatTensor)
                if torch.cuda.is_available():
                    img_fra, img_sig = Variable(img_fra.cuda(), requires_grad=False), Variable(img_sig.cuda(),
                                                                                           requires_grad=False)
                    label = Variable(label.cuda(), requires_grad=False)

                optimizer.zero_grad()
                res_pre = net(img_sig, img_fra)
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

            net.eval()
            net.mode = 'val'
            with torch.no_grad():
                for i, data in enumerate(val_dataloader):
                    img_fra, img_sig, label = data['image_fra'], data['image_sig'], data['label']
                    img_fra = img_fra.type(torch.FloatTensor)
                    img_sig = img_sig.type(torch.FloatTensor)
                    label = label.type(torch.FloatTensor)
                    if torch.cuda.is_available():
                        img_fra, img_sig = Variable(img_fra.cuda(), requires_grad=False), Variable(img_sig.cuda(),
                                                                                           requires_grad=False)
                        label = Variable(label.cuda(), requires_grad=False)
                    res_pre = net(img_sig, img_fra)
                    loss = fun_loss(res_pre, label)
                    val_loss += loss

                    # 计算准确率和混淆矩阵
                    for m in range(res_pre.shape[0]):
                        label_pre.append(class_num[torch.argmax(res_pre[m])])
                        label_true.append(class_num[torch.argmax(label[m])])
                    for j in range(res_pre.shape[0]):
                        if torch.argmax(res_pre[j]) == torch.argmax(label[j]):
                            epoch_val_acc = epoch_val_acc + 1

                # 保存模型
                if epoch_val_acc / len(val_fold) >= best_val_acc:
                    best_val_acc = epoch_val_acc / len(val_fold)
                    torch.save(net.state_dict(), './models_save/time_fra_cbam_vit_k_d1_'+str(k)+'.pth')

                label_pre = label_binarize(label_pre, classes=[i for i in range(7)])
                label_true = label_binarize(label_true, classes=[i for i in range(7)])
                precision = precision_score(label_true, label_pre, average='macro')
                recall = recall_score(label_true, label_pre, average='macro')

                val_precision.append(precision)
                val_recall.append(recall)
                train_acc.append(epoch_train_acc / len(train_fold))
                val_acc.append(epoch_val_acc / len(val_fold))
                train_total_loss.append(train_loss / (len(train_fold) / train_batch))
                val_total_loss.append(val_loss / len(val_fold) / train_batch)
            print('train loss：{}'.format(train_loss / math.ceil((len(train_fold) / train_batch))))
            print('val loss：{}'.format(val_loss / math.ceil((len(val_fold) / train_batch))))
            print('train acc：{}'.format(epoch_train_acc / len(train_fold)))
            print('val acc：{}'.format(epoch_val_acc / len(val_fold)))
            print("Precision score:", precision)
            print("Recall score:", recall)
        print('training end，the best acc on trainset：{}, the best acc on valset: {}'.format(best_train_acc, best_val_acc))

    # 准确率曲线图
    plt.ylim(0, 1)
    show_data1 = train_acc
    show_data2 = val_acc
    x_data = list(range(1, len(show_data1) + 1))
    ln1, = plt.plot(x_data, show_data1, color='blue', linewidth=2.0, linestyle='--')
    ln2, = plt.plot(x_data, show_data2, color='red', linewidth=2.0, linestyle='-.')
    plt.legend(handles=[ln1, ln2], labels=['train_acc', 'val_acc'])
    plt.title('model_supervised')
    plt.savefig('./results/time_fra_cbam_vit_k_d1_Acc.png')
    plt.show()

    show_data1 = train_total_loss
    show_data2 = val_total_loss
    x_data = list(range(1, len(show_data1) + 1))
    for i in range(len(show_data1)):
        show_data1[i] = show_data1[i].cpu().detach().numpy()
    ln1, = plt.plot(x_data, show_data1, color='blue', linewidth=2.0, linestyle='--')
    for i in range(len(show_data2)):
        show_data2[i] = show_data2[i].cpu().detach().numpy()
    ln2, = plt.plot(x_data, show_data2, color='red', linewidth=2.0, linestyle='--')
    plt.legend(handles=[ln1, ln2], labels=['train_loss', 'val_loss'])
    plt.title('model_supervised')
    plt.savefig('./results/time_fra_cbam_vit_k_d1_loss.png')
    plt.show()

    # 绘制精确率、召回率、F1值
    plt.figure()
    plt.ylim(0, 1)
    show_data1 = val_precision
    show_data2 = val_recall
    x_data = list(range(1, len(show_data1) + 1))
    plt.subplot(1, 2, 1)
    plt.plot(x_data, show_data1, color='blue', linewidth=2.0, linestyle='-')
    plt.title('model_cwruk_Precision')

    plt.subplot(1, 2, 2)
    plt.plot(x_data, show_data2, color='blue', linewidth=2.0, linestyle='-')
    plt.title('model_supervised_Recall')

    plt.savefig('./results/time_fra_cbam_vit_k_d1_PR.png')
    plt.show()
    # 绘制ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(7):
        fpr[i], tpr[i], _ = roc_curve(label_true[:, i], label_pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr['micro'], tpr['micro'], _ = roc_curve(label_true.ravel(), label_pre.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(7)]))
    mean_tpr = np.zeros_like(all_fpr)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 2
    plt.figure()

    colors = cycle(
        ['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue', 'yellow', 'purple', 'tomato', 'deepskyblue', 'pink'])
    for i, color in zip(range(7), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('model_supervised_ROC')
    plt.legend(loc="lower right")
    plt.savefig("./results/time_fra_cbam_vit_k_d1_ROC.png")
    plt.show()


if __name__ == '__main__':
    data_train, label_train, data_test, label_test = load_data()
    print('*' * 10)
    print('训练集数量：', len(label_train))
    print('测试集数量：', len(label_test))

    train_batch = 32
    num_epoch = 50

    train_dataset = new_method_dataset(data=data_train, label=label_train)

    net = fra_sig_network(vit_weight='./models_save/con_vit_vir.pth', mode='train', num_out=7)
    # net = time_model(mode='train', num_out=7)
    if torch.cuda.is_available():
        net.cuda()

    fun_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

    train()
