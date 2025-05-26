# 1.导入所需要的库
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image

# 2.定义超参数
Batch_size = 128    #每次迭代输入模型的数量为128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#用GPU或CPU训练
epochs = 10       #训练的次数

# 3.图像预处理管道
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 4.下载并加载数据集
from torch.utils.data import DataLoader

train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=Batch_size, shuffle=True)

# 5.定义网络结构
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 6.初始化模型和优化器
model = Digit().to(device)
optimizer = optim.Adam(model.parameters())


# 7.训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 3000 == 0:
            print("第{}次训练:loss: {:.3f}".format(epoch, loss.item()))


# 8.测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("测试：loss:{:.3f},正确率:{:.2f}\n".format(test_loss, 100.0 * correct / len(test_loader.dataset)))
'''
# 9.训练流程
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

#保存训练好的模型方便之后的测试
model_save_path = "./mnist_cnn.pth"
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),},model_save_path)
print(f"模型已成功保存到 {model_save_path}")
'''
# 10.测试自己数据
def test_mydata():
    #加载已保存的模型
    checkpoint = torch.load("./mnist_cnn.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 图像预处理
    image = Image.open('mytest/img_7.png').convert('L')
    image = image.resize((28, 28))

    #mage=image.point(lambda p:255-p)#像素值反转

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        output = model(image_tensor)
    prob, pred = torch.exp(output).max(dim=1)
    print(f'预测结果：{pred.item()} 准确率：{prob.item():.2%}')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    img_show = image_tensor.cpu().squeeze()
    img_show = img_show * 0.3081 + 0.1307  # 反归一化
    plt.imshow(img_show.numpy(), cmap='gray')
    plt.title(f'预测结果：{pred.item()} 准确率：{prob.item():.2%}')
    plt.show()

if __name__ == '__main__':
    test_mydata()