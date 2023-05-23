from glob import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models, utils
from PIL import Image
from torchsummary import summary
from model import Generator, Discriminator

# Hyper parameters
BATCH_SIZE = 8
LEARNING_RATE_G = 2e-3
LEARNING_RATE_D = 1e-3
LAMBDA = 100
EPOCHS = 50
TIMER = 2
LOSS_THRESHOLD = 0.02

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, line_path, sketch_path='empty', resize=-1):  # resize 表示训练图片裁剪后保留的大小， 默认（-1）为不裁剪
        self.photo_size = resize
        self.data = self.init_dataset(line_path, self.photo_size)
        if sketch_path != 'empty':
            self.label = self.init_dataset(sketch_path, self.photo_size)
        else:
            self.label = self.data

    def init_dataset(self, data_path, resize):
        if resize == -1:
            compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
        else:
            compose = transforms.Compose([transforms.Resize([resize, resize]), transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

        photo_names = glob(os.path.join(data_path, '*.jpg'))
        new_dataset = []
        for (i, name) in enumerate(photo_names):
            photo = Image.open(name)
            new_dataset.append(compose(photo))
        return new_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


# 获得测试图片的编号，输出时要用
def get_number(string):
    num = 0
    for idx in range(len(string)):
        if '0' <= string[idx] <= '9':
            while '0' <= string[idx] <= '9':
                num = num * 10 + int(string[idx])
                idx += 1
            break
    return num

line_path = r'.\CGI-PSG-Training_Set\train_line'
sketch_path = r'.\CGI-PSG-Training_Set\train_sketch'
dataset = MyDataset(line_path=line_path, sketch_path=sketch_path, resize=256)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # 处理训练集
# 定义生成器、判别器及对应的优化器，以及损失函数
G = Generator().cuda()
D = Discriminator().cuda()
optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
# 损失函数
Loss_G = []
Loss_D = [1.00]
loss_bce = nn.BCELoss()
loss_l1 = nn.L1Loss()

for epoch in range(EPOCHS):
    epoch_loss_G = 0
    epoch_loss_D = 0
    batch_num = len(data_loader)
    for lines, sketches in data_loader:
        lines = lines.cuda()
        sketches = sketches.cuda()

        output_G = G(lines)
        # 先训练D
        if (epoch + TIMER) % TIMER == 0 and (Loss_D[-1] > LOSS_THRESHOLD):
            optimizer_D.zero_grad()
            real_output_D = D(lines, sketches)
            real_loss_D = loss_bce(real_output_D, torch.ones_like(real_output_D, device='cuda'))
            real_loss_D.backward()
            fake_output_D = D(lines, output_G.detach())
            fake_loss_D = loss_bce(fake_output_D, torch.zeros_like(fake_output_D, device='cuda'))
            fake_loss_D.backward()
            loss_D = real_loss_D + fake_loss_D
            optimizer_D.step()

        # 再训练G
        optimizer_G.zero_grad()
        output_D = D(lines, output_G)
        BCE_loss_G = loss_bce(output_D, torch.ones_like(output_D, device='cuda'))
        L1_loss_G = loss_l1(output_G, sketches)
        loss_G = 0.1 * (BCE_loss_G + LAMBDA * L1_loss_G)
        loss_G.backward()
        optimizer_G.step()

        with torch.no_grad():
            epoch_loss_G += loss_G.item()
            if (epoch + TIMER) % TIMER == 0:
                epoch_loss_D += loss_D.item()
    # 求平均损失
    with torch.no_grad():
        epoch_loss_G /= batch_num
        if (epoch + TIMER) % TIMER == 0:
            epoch_loss_D /= batch_num
        else:
            epoch_loss_D = Loss_D[-1]
        Loss_G.append(epoch_loss_G)
        Loss_D.append(epoch_loss_D)
        print('Epoch: ', epoch, '  Loss_G: ', epoch_loss_G, '  Loss_D: ', epoch_loss_D)

# 输出结果
test_path = r'.\CGI-PSG-Testing_Set'
photo_names = glob(os.path.join(test_path, '*.jpg'))
compose = transforms.Compose([transforms.Resize([256, 256]), transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
for (i, name) in enumerate(photo_names):
    line = compose(Image.open(name)).view(1, 3, 256, 256).cuda()
    generated_sketches = G(line)
    number = get_number(name)
    unloader = transforms.Compose([transforms.ToPILImage(), transforms.Resize([1024, 1024])])
    img = unloader(generated_sketches.view(3, 256, 256))
    L = img.convert('L')
    img = L.convert('RGB')
    save_path = '.\\Result\\' + str(number) + '.jpg'
    img.save(save_path)


