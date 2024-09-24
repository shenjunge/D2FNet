import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import new_method_dataset, time_dataset, fra_dataset
from new_model import fra_sig_network, p_net
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, f1_score
from itertools import cycle
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from draw_matrix import draw_confusion_matrix
import torchvision
from sklearn.manifold import TSNE
import torch.nn.functional as F
matplotlib.use('TkAgg')
from tsne import main
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from vit_model_cls import VisionTransformer

def tranfer_data():
    data = []
    label = []
    for file_name in os.listdir('./data/vir/C_imgs'):
        data.append('./data/vir/C_imgs/' + file_name)
        label.append(int(file_name[0]))
    return data, label

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


def test():
    print('*' * 5 + 'start testing' + '*' * 5)
    class_num = [0, 1, 2, 3, 4, 5, 6]
    test_acc = 0
    label_pre1 = []
    label_true = []

    net.eval()
    net.mode = 'test'
    for i, data in enumerate(test_dataloader):
        img_fra, img_sig, label = data['image_fra'], data['image_sig'], data['label']
        # img_sig, label = data['image_sig'], data['label']
        img_fra = img_fra.type(torch.FloatTensor)
        img_sig = img_sig.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        if torch.cuda.is_available():
            img_fra, img_sig = Variable(img_fra.cuda(), requires_grad=False), Variable(img_sig.cuda(),
                                                                                           requires_grad=False)
            img_sig = Variable(img_sig.cuda(), requires_grad=False)
            label = Variable(label.cuda(), requires_grad=False)
        res_pre = net(img_sig, img_fra)
        # res_pre = net(img_sig)

        # 画聚类图
        feature = res_pre.cpu().detach().numpy()
        np.save('./tsne/time_shuffle/data{}.npy'.format(i), feature)
        '''result = tsne.fit_transform(feature)
        ax.plot3D(result[:, 0], result[:, 1], result[:, 2])
        plt.scatter(result[:, 0], result[:, 1], result[:, 2], label='t-SNE')'''
        '''for i in range(result.shape[0]):
            plt.text(result[i, 0], result[i, 1], str(np.argmax(label[i].cpu().detach().numpy())),
                     color=plt.cm.Set1(np.argmax(label[i].cpu().detach().numpy()) / 10.),
                     fontdict={'weight': 'bold', 'size': 9})'''

        # 计算准确率和混淆矩阵
        for m in range(res_pre.shape[0]):
            label_pre1.append(class_num[torch.argmax(res_pre[m])])
            label_true.append(class_num[torch.argmax(label[m])])
        for j in range(res_pre.shape[0]):
            if torch.argmax(res_pre[j]) == torch.argmax(label[j]):
                test_acc = test_acc + 1

    label_pre = label_binarize(label_pre1, classes=[i for i in range(7)])
    label_true = label_binarize(label_true, classes=[i for i in range(7)])
    precision = precision_score(label_true, label_pre, average='macro')
    recall = recall_score(label_true, label_pre, average='macro')

    print('本轮测试集准确率：{}'.format(test_acc / len(data_test)))
    print("Precision_score:", precision)
    print("Recall_score:", recall)
    draw_confusion_matrix(label_test, label_pre1,
                          ['health', 'fatigue out', 'fatigue in', 'drilled out', 'mee out', 'mee in', 'emd out'],
                          title=' ',
                          pdf_save_path='./time_shuffle_matrix.png',
                          dpi=300)

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
    plt.savefig("./time_shuffle_ROC.png")
    plt.show()


if __name__ == '__main__':
    data_train, label_train, data_test, label_test = load_data()
    # data_test, label_test = tranfer_data()
    print('*' * 10)
    # print('训练集数量：', len(label_train))
    print('测试集数量：', len(label_test))

    train_batch = 32

    test_dataset = new_method_dataset(data=data_test, label=label_test)
    test_dataloader = DataLoader(test_dataset, batch_size=train_batch, shuffle=False, num_workers=11)

    net = fra_sig_network(vit_weight='./models_save/con_vit_vir.pth', mode='test', num_out=7)
    '''net = torchvision.models.shufflenet_v2_x0_5()
    net.conv1[0] = torch.nn.Conv2d(1, 24, (3, 3), (2, 2), (1, 1), bias=False)
    net.fc = torch.nn.Linear(1024, 7)'''
    # 加载参数
    save_model = torch.load('./models_save/pu/消融实验/时域图+cbam+vit+k折/time_fra_cbam_vit_k_1.pth')
    '''model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)'''
    # net.load_state_dict(save_model, strict=False)
    if torch.cuda.is_available():
        net.cuda()

    test()
