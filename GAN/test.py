import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设定权重参数路径
checkpoint_path = f'.\GAN_checkpoint\model_epoch_100.pth'

# 读取train_list.txt文件
data_list_file = "./dataset/train_list.txt"
with open(data_list_file, "r") as f:
    lines = f.readlines()

file_paths, labels = zip(*[line.strip().split() for line in lines])# 将文件路径和标签分开
labels = [int(label) for label in labels]
file_paths = ['' + file_path for file_path in file_paths]
data = list(zip(file_paths, labels))# 合并文件路径和标签
random.shuffle(data)# 打乱数据

# 划分训练集和验证集（80%训练集，20%验证集）
split_index = int(0.8 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

# 定义自定义数据集类
class Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# 定义数据转换
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建训练集和验证集的数据集实例
train_dataset = Dataset(train_data, transform=train_transform)
val_dataset = Dataset(val_data, transform=test_transform)

# 创建训练集和验证集的数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 获取一批数据
batch_data, batch_labels = next(iter(train_loader))

# 可视化数据
def imshow(img, title):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.show()

# 随机选择一张图像进行可视化
index = random.randint(0, len(batch_data) - 1)
img, label = batch_data[index], batch_labels[index]
classname = str(label)

# 可视化图像
imshow(img, title=classname)

# 设置随机种子
random.seed(42)
torch.manual_seed(42)

# 定义生成器和鉴别器，并将它们移动到 GPU 上
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=3, img_height=64, img_width=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.img_height = img_height
        self.img_width = img_width

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, num_channels, 4, 2, 1, bias=False),  # 输出通道数修改为 num_channels
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(256 * 14 * 14 * 4, 12)  # Adjust the output size for 12 classes
        self.fc2 = nn.Linear(256*16, 12)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        if(x.shape[2] == 4):
            x = x.view(32, -1)
            x = self.fc2(x)
        else:
            x = x.view(-1, 256 * 14 * 14 * 4)
            x = self.fc1(x)
        return x


# 实例化生成器和鉴别器，并将它们移动到 GPU 上
generator = Generator(latent_dim=100).to(device)
discriminator = Discriminator().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

checkpoint = torch.load(checkpoint_path)
epoch = checkpoint['epoch']
discriminator.load_state_dict(checkpoint['model_state_dict'])
optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']


latent_dim = 100
num_epochs = 100
counter = 0

# 在验证集上评估模型
discriminator.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = discriminator(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy * 100:.2f}%')