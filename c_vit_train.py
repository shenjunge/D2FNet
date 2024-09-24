import torch.cuda
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_augment import cons_dataset
from Network import cons_vit
from con_loss import ContrastiveLoss


def load_data():
    label_train = []
    label_test = []
    with open('./data/CWRU/images/train.txt', 'r') as f:
        line_train = f.readlines()
    with open('./data/CWRU/images/test.txt', 'r') as f:
        line_test = f.readlines()
    for name in line_train:
        label_train.append(int(name.split('/')[-1][0]))
    for name in line_test:
        label_test.append(int(name.split('/')[-1][0]))
    return line_train, label_train, line_test, label_test


def train():
    print('-' * 5 + 'start training' + '-' * 5)
    min_loss = 100
    for epoch in range(epoch_num):
        train_loss = 0
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=10)
        print('-' * 10 + 'epoch ' + str(epoch + 1) + '/' + str(epoch_num) + '-' * 10)

        # 训练训练集
        contractive_vit.train()
        for i, data in enumerate(train_dataloader):
            trans1_img, trans2_img, labels = data['image_trans1'], data['image_trans2'], data['label']
            inputs1 = trans1_img.type(torch.FloatTensor)
            inputs2 = trans2_img.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs1, inputs2, labels_v = Variable(inputs1.cuda(), requires_grad=False), \
                                             Variable(inputs2.cuda(), requires_grad=False), \
                                             Variable(labels.cuda(), requires_grad=False)
            optimizer.zero_grad()

            result1 = contractive_vit(inputs1)
            result2 = contractive_vit(inputs2)

            # 计算损失
            batch = inputs1.shape[0]
            con_loss = ContrastiveLoss(batch_size=batch)
            loss = con_loss(result1, result2)
            train_loss += loss
            loss.backward()
            optimizer.step()
        print('本轮损失值：{}'.format(train_loss / (len(train_data) / inputs2.shape[0])))

        if (train_loss / (len(train_data) / inputs1.shape[0])) < min_loss:
            min_loss = train_loss / (len(train_data) / inputs2.shape[0])
            torch.save(contractive_vit.state_dict(), './models_save/con_vit_cwru.pth')


if __name__ == '__main__':
    train_data, train_label, val_data, val_label = load_data()
    print('*' * 10)
    print('训练集数量：', len(train_label))
    print('测试集数量：', len(val_label))

    # 创建数据集
    batch_size_train = 32
    epoch_num = 500

    train_dataset = cons_dataset(
        fault_data=train_data,
        fault_label=train_label)

    # 模型初始化
    contractive_vit = cons_vit()
    if torch.cuda.is_available():
        contractive_vit.cuda()

    weights_dict = torch.load('./vit_base_patch16_224.pth', map_location='cuda:0')
    # 删除不需要的权重
    del_keys = ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    contractive_vit.load_state_dict(weights_dict, strict=False)

    # 优化器
    optimizer = torch.optim.Adam(contractive_vit.parameters(), lr=0.001, weight_decay=0.001)

    # 训练
    train()
