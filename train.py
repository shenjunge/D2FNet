import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import new_method_dataset, time_dataset, fra_dataset, orignal_dataset
from new_model import fra_sig_network, p_net, orignal_network
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, f1_score
from itertools import cycle
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import torchvision
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
    total_loss = []
    test_loss = []
    val_precision = []
    val_recall = []
    best_train_acc = 0
    best_test_acc = 0

    for epoch in range(num_epoch):
        print('-' * 10 + 'epoch ' + str(epoch + 1) + '/' + str(num_epoch) + '-' * 10)
        epoch_train_acc = 0
        epoch_test_acc = 0
        label_pre = []
        label_true = []
        epoch_loss = 0
        epoch_test_loss = 0

        net.train()
        net.mode = True
        for i, data in enumerate(train_dataloader):
            batch_acc = 0
            img_fra, img_sig, label = data['image_fra'], data['image_sig'], data['label']
            # img_sig, label = data['image_fra'], data['label']
            img_fra = img_fra.type(torch.FloatTensor)
            img_sig = img_sig.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            if torch.cuda.is_available():
                img_fra, img_sig = Variable(img_fra.cuda(), requires_grad=False), Variable(img_sig.cuda(),
                                                                                           requires_grad=False)
                # img_sig = Variable(img_sig.cuda(), requires_grad=False),
                label = Variable(label.cuda(), requires_grad=False)

            optimizer.zero_grad()
            res_pre = net(img_sig, img_fra)
            # res_pre = net(img_sig[0])
            loss = fun_loss(res_pre, label)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            # 计算准确率
            for j in range(res_pre.shape[0]):
                if torch.argmax(res_pre[j]) == torch.argmax(label[j]):
                    epoch_train_acc = epoch_train_acc + 1
                    batch_acc += 1
            # print('{}/{}:{}'.format((i + 1) * train_batch, len(data_train), batch_acc / img_sig.shape[0]))

        net.eval()
        net.mode = False
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                img_fra, img_sig, label = data['image_fra'], data['image_sig'], data['label']
                # img_sig, label = data['image_fra'], data['label']
                img_fra = img_fra.type(torch.FloatTensor)
                img_sig = img_sig.type(torch.FloatTensor)
                label = label.type(torch.FloatTensor)
                if torch.cuda.is_available():
                    img_fra, img_sig = Variable(img_fra.cuda(), requires_grad=False), Variable(img_sig.cuda(),
                                                                                               requires_grad=False)
                    # img_sig = Variable(img_sig.cuda(), requires_grad=False)
                    label = Variable(label.cuda(), requires_grad=False)
                res_pre = net(img_sig, img_fra)
                # res_pre = net(img_sig)
                loss = fun_loss(res_pre, label)
                epoch_test_loss += loss

                # 计算准确率和混淆矩阵
                label_pre.append(class_num[torch.argmax(res_pre[0])])
                label_true.append(class_num[torch.argmax(label[0])])
                for j in range(res_pre.shape[0]):
                    if torch.argmax(res_pre[j]) == torch.argmax(label[j]):
                        epoch_test_acc = epoch_test_acc + 1

            # 保存模型
            # if epoch_test_acc / len(data_test) >= best_test_acc:
            best_train_acc = epoch_train_acc / len(data_train)
            best_test_acc = epoch_test_acc / len(data_test)
            torch.save(net.state_dict(), './models_save/p_net/pu_fra{}.pth'.format(epoch))

            label_pre = label_binarize(label_pre, classes=[i for i in range(7)])
            label_true = label_binarize(label_true, classes=[i for i in range(7)])
            precision = precision_score(label_true, label_pre, average='macro')
            recall = recall_score(label_true, label_pre, average='macro')

            val_precision.append(precision)
            val_recall.append(recall)
            train_acc.append(epoch_train_acc / len(data_train))
            val_acc.append(epoch_test_acc / len(data_test))
            total_loss.append(epoch_loss / (len(data_train) / train_batch))
            test_loss.append(epoch_test_loss/ len(data_test) / train_batch)
            print('本轮训练损失值：{}'.format(epoch_loss / (len(data_train) / train_batch)))
            print('本轮测试损失值：{}'.format(epoch_test_loss / (len(data_test) / train_batch)))
            print('本轮训练集准确率：{}'.format(epoch_train_acc / len(data_train)))
            print('本轮测试集准确率：{}'.format(epoch_test_acc / len(data_test)))
            print("Precision_score:", precision)
            print("Recall_score:", recall)
    print('训练结束，训练集最佳准确率为：{}'.format(best_train_acc))
    print('训练结束，测试集最佳准确率为：{}'.format(best_test_acc))

    # 准确率曲线图
    plt.ylim(0, 1)
    show_data1 = train_acc
    show_data2 = val_acc
    x_data = list(range(1, len(show_data1) + 1))
    ln1, = plt.plot(x_data, show_data1, color='blue', linewidth=2.0, linestyle='--')
    ln2, = plt.plot(x_data, show_data2, color='red', linewidth=2.0, linestyle='-.')
    plt.legend(handles=[ln1, ln2], labels=['train_acc', 'val_acc'])
    plt.title('time_fra_cbam')
    plt.savefig('./results/p_net_acc.png')
    plt.show()

    show_data1 = total_loss
    show_data2 = test_loss
    x_data = list(range(1, len(show_data1) + 1))
    for i in range(len(show_data1)):
        show_data1[i] = show_data1[i].cpu().detach().numpy()
    for i in range(len(show_data2)):
        show_data2[i] = show_data2[i].cpu().detach().numpy()
    ln1, = plt.plot(x_data, show_data1, color='blue', linewidth=2.0, linestyle='--')
    ln2, = plt.plot(x_data, show_data2, color='red', linewidth=2.0, linestyle='--')
    plt.legend(handles=[ln1, ln2], labels=['train_loss'])
    plt.title('time_fra_cbam')
    plt.savefig('./results/p_net_loss.png')
    plt.show()

    # 绘制精确率、召回率、F1值
    plt.figure()
    plt.ylim(0, 1)
    show_data1 = val_precision
    show_data2 = val_recall
    x_data = list(range(1, len(show_data1) + 1))
    plt.subplot(1, 2, 1)
    plt.plot(x_data, show_data1, color='blue', linewidth=2.0, linestyle='-')
    plt.title('p-net_Precision')

    plt.subplot(1, 2, 2)
    plt.plot(x_data, show_data2, color='blue', linewidth=2.0, linestyle='-')
    plt.title('p-net_Recall')

    plt.savefig('./results/p_net_PR.png')
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
    plt.title('time_fra_cbam_ROC')
    plt.legend(loc="lower right")
    plt.savefig("./results/p_net_ROC.png")
    plt.show()


if __name__ == '__main__':
    data_train, label_train, data_test, label_test = load_data()
    print('*' * 10)
    print('训练集数量：', len(label_train))
    print('测试集数量：', len(label_test))

    train_batch = 32
    num_epoch = 50

    train_dataset = orignal_dataset(data=data_train, label=label_train)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=11)

    test_dataset = orignal_dataset(data=data_test, label=label_test)
    test_dataloader = DataLoader(test_dataset, batch_size=train_batch, shuffle=False, num_workers=11)

    # net = p_net(vit_weight='./models_save/con_vit_vir.pth', mode='train', num_out=7)
    # net = p_net(mode=True)
    net = orignal_network()
    if torch.cuda.is_available():
        net.cuda()

    fun_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

    train()
