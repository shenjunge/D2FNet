import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from torchvision import transforms


class fra_dataset(Dataset):
    def __init__(self, data, label):
        super(fra_dataset, self).__init__()
        self.fault_data = data
        self.fault_label = label
        self.transform2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((224, 224)),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.fault_label)

    def __getitem__(self, item):
        # 制作标签
        label = np.asarray([0, 0, 0, 0, 0, 0, 0])
        label[self.fault_label[item]] = 1

        # 读取频率图片
        image_fra = Image.open(self.fault_data[item][:-1])
        # image_fra = Image.open(self.fault_data[item])
        image_fra = self.transform2(image_fra)

        return {'imidx': item, 'image_fra': image_fra, 'label': label}


class time_dataset(Dataset):
    def __init__(self, data, label):
        super(time_dataset, self).__init__()
        self.fault_data = data
        self.fault_label = label
        self.transform1 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((64, 64))])

    def __len__(self):
        return len(self.fault_label)

    def __getitem__(self, item):
        # 制作标签
        label = np.asarray([0, 0, 0, 0, 0, 0, 0])
        label[self.fault_label[item]] = 1

        # 制作信号图片
        image_path = self.fault_data[item][:-1]
        signal_path = (image_path.replace('images', 'signals')).replace('png', 'npy')
        signal_data = np.load(signal_path)[0]
        p_signal_data = ((signal_data-min(signal_data)) / (max(signal_data)-min(signal_data))) * 2 - 1
        image_sig = p_signal_data.reshape(64, 64)
        image_sig = self.transform1(image_sig)

        return {'imidx': item, 'image_sig': image_sig, 'label': label}


class new_method_dataset(Dataset):
    def __init__(self, data, label):
        super(new_method_dataset, self).__init__()
        self.fault_data = data
        self.fault_label = label
        self.transform1 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((64, 64))])
        self.transform2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((224, 224)),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.fault_label)

    def __getitem__(self, item):
        # 制作标签
        label = np.asarray([0, 0, 0, 0, 0, 0, 0])
        label[self.fault_label[item]] = 1

        # 读取频率图片
        image_fra = Image.open(self.fault_data[item][:-1])
        # image_fra = Image.open(self.fault_data[item])
        image_fra = self.transform2(image_fra)

        # 制作信号图片
        image_path = self.fault_data[item][:-1]
        # image_path = self.fault_data[item]
        signal_path = (image_path.replace('images', 'signals')).replace('png', 'npy')
        # signal_path = (image_path.replace('C_imgs', 'C_sigs')).replace('png', 'npy')
        signal_data = np.load(signal_path)[0]
        p_signal_data = ((signal_data-min(signal_data)) / (max(signal_data)-min(signal_data))) * 2 - 1
        image_sig = p_signal_data.reshape(64, 64)
        image_sig = self.transform1(image_sig)

        return {'imidx': item, 'image_fra': image_fra, 'image_sig': image_sig, 'label': label}


class orignal_dataset(Dataset):
    def __init__(self, data, label):
        super(orignal_dataset, self).__init__()
        self.fault_data = data
        self.fault_label = label
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((224, 224)),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.fault_label)

    def __getitem__(self, item):
        # 制作标签
        label = np.asarray([0, 0, 0, 0, 0, 0, 0])
        label[self.fault_label[item]] = 1

        # 读取频率图片
        image_fra = Image.open(self.fault_data[item][:-1])
        # image_fra = Image.open(self.fault_data[item])
        image_fra = self.transform2(image_fra)

        # 制作信号图片
        image_path = self.fault_data[item][:-1]
        # image_path = self.fault_data[item]
        signal_path = (image_path.replace('images', 'signals')).replace('png', 'npy')
        # signal_path = (image_path.replace('C_imgs', 'C_sigs')).replace('png', 'npy')
        signal_data = np.load(signal_path)[0]
        p_signal_data = ((signal_data-min(signal_data)) / (max(signal_data)-min(signal_data))) * 2 - 1
        image_sig = torch.from_numpy(p_signal_data)
        signal_data = torch.unsqueeze(image_sig, dim=0)
        # image_sig = p_signal_data.reshape(64, 64)


        return {'imidx': item, 'image_fra': image_fra, 'image_sig': signal_data, 'label': label}


class fault_net_dataset(Dataset):
    def __init__(self, data, label):
        super(fault_net_dataset, self).__init__()
        self.fault_data = data
        self.fault_label = label
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((224, 224)),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.fault_label)

    def __getitem__(self, item):
        # 制作标签
        label = np.asarray([0, 0, 0, 0, 0, 0, 0])
        label[self.fault_label[item]] = 1

        # 制作信号图片
        image_path = self.fault_data[item][:-1]
        signal_path = (image_path.replace('images', 'signals')).replace('png', 'npy')
        signal_data = np.load(signal_path)[0]
        p_signal_data = ((signal_data-min(signal_data)) / (max(signal_data)-min(signal_data))) * 2 - 1
        image_sig = p_signal_data.reshape(64, 64)
        image_sig1 = self.transform1(image_sig)

        signal_path = ((image_path.replace('vir', 'current')).replace('png', 'npy')).replace('images', '1')
        signal_data = np.load(signal_path)[0]
        p_signal_data = ((signal_data - min(signal_data)) / (max(signal_data) - min(signal_data))) * 2 - 1
        image_sig = p_signal_data.reshape(64, 64)
        image_sig2 = self.transform1(image_sig)

        signal_path = ((image_path.replace('vir', 'current')).replace('png', 'npy')).replace('images', '1')
        signal_data = np.load(signal_path)[0]
        p_signal_data = ((signal_data - min(signal_data)) / (max(signal_data) - min(signal_data))) * 2 - 1
        image_sig = p_signal_data.reshape(64, 64)
        image_sig3 = self.transform1(image_sig)

        image_sig = torch.cat([image_sig1, image_sig2, image_sig3], dim=0)
        return {'imidx': item, 'image_sig': image_sig, 'label': label}
