# 基于MNIST数字手写体识别
## 简介
一个基于PyTorch实现的卷积神经网络，用于MNIST手写数字识别，包含完整训练流程和实际测试示例，也可以利用训练好的模型去测试自定义手写数字，正确率也非常高。

### 环境配置
- Python 3.8
- Pytorch 2.3
- torchvision 0.18.0
- matplotlib 3.7.5

## 一、导入需要的库
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image
```
### 1.1 torch：PyTorch 的核心库，提供张量（Tensor）计算和 GPU 加速支持。
- 张量操作（如矩阵乘法等）。
- 自动微分（autograd，用于神经网络训练）。
- 设备管理（CPU/GPU 切换，如 .to('cuda')）。
### 1.2 torchvision：PyTorch 的计算机视觉工具库。
子模块：
- datasets：预加载常用数据集（如 MNIST、CIFAR-10）。
- transforms：图像预处理（如裁剪、归一化、数据增强）。
- models：预训练模型（如 ResNet、VGG）。

## 二、MINST数据集下载
运行程序会自动下载并存放在\data目录下。
```python
from torch.utils.data import DataLoader
train_set = datasets.MNIST("data", train=True, download=True,transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)
```

### 2.1 数据预处理管道
```python
pipeline = transforms.Compose([
    transforms.ToTensor(),          # 转换为张量（0-1范围）
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])
```

### 2.2 创建数据加载器
```python
train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=Batch_size, shuffle=True)
```

## 三、CNN网络结构
```python
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1：1输入通道 → 10输出通道，5x5卷积核
        self.conv1 = nn.Conv2d(1, 10, 5)
        # 卷积层2：10 → 20通道，3x3卷积核
        self.conv2 = nn.Conv2d(10, 20, 3)
        # 全连接层1：20*10*10 → 500节点
        self.fc1 = nn.Linear(20*10*10, 500)
        # 全连接层2：500 → 10输出（对应0-9数字）
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Conv1 → ReLU → MaxPool（2x2）
        x = F.relu(F.max_pool2d(self.conv1(x), 2)  
        # Conv2 → ReLU（无池化）
        x = F.relu(self.conv2(x))                 
        # 展平特征图
        x = x.view(x.size(0), -1)                 
        # FC1 → ReLU
        x = F.relu(self.fc1(x))
        # 最终输出
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)            # 对数概率输出
```
本项目使用的激活函数为Relu激活函数，在此简单介绍一下Relu激活函数：
Relu的全称为修正线性单元(Rectified Linear Unit) ，其函数表达式和图像如下所示：
$$
Relu=\begin{cases}
    x,x>0\\0,x<=0
\end{cases}
$$

### Relu的优点如下：
由Relu的原始图像和导数图像可知，Relu可能使部分神经元的值变为0,降低神经网络复杂性,从而有效缓解过拟合的问题。由于当x>0时，Relu的梯度恒为1，所以随着神经网络越来越复杂，不会导致梯度累乘后变得很大或很小，从而不会发生梯度爆炸或梯度消失问题。Relu的计算非常简单，提高了神经网络的效率

## 四、训练过程
```python
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
```


### 训练过程步骤

| 步骤               | 说明                                                                 |
|--------------------|----------------------------------------------------------------------|
| **1. 获取训练数据及标签** | 从数据加载器（DataLoader）中读取输入数据和对应标签 。         |
| **2. 梯度清零**         | 清除优化器中上一轮的梯度，避免梯度累积。                             |
| **3. 模型预测**         | 输入数据通过模型前向传播，得到预测值 `pred`。                        |
| **4. 计算损失**         | 比较预测值  和真实标签 ，计算损失值 `loss`。                |
| **5. 反向传播**         | 根据损失计算模型参数的梯度。                                         |
| **6. 更新参数**         | 优化器根据梯度更新模型参数。                                         |
| **7. 保存模型**         | 定期保存模型权重（如每N个epoch）。                                   |
| **8. 保存优化器**       | 可选：保存优化器状态以便恢复训练。                                   |
| **9. 打印损失**         | 监控训练过程，打印当前批次或epoch的损失。                            |







## 五、测试过程
```python
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
```

### 测试过程步骤

| 步骤                     | 说明                                                                 |
|--------------------------|----------------------------------------------------------------------|
| **1. 加载模型和优化器**   | 从文件加载训练好的模型权重（和优化器状态）。                         |
| **2. 设置模型为评估模式** | 关闭Dropout、BatchNorm等训练专用层的行为。                           |
| **3. 禁用梯度计算**       | 减少内存占用，加速推理。                                             |
| **4. 获取测试数据**       | 从测试集读取输入数据（无需标签，除非需计算指标）。                   |
| **5. 模型前向传播**       | 输入数据通过模型，得到预测输出。                                     |
| **6. 计算评估指标**       | 可选：若有标签，计算准确率、F1等指标。                               |
| **7. 后处理与输出结果**   | 对输出进行解析（如分类取argmax），打印或保存结果。                   |
| **8. 可视化结果**         | 可选：绘制预测图像、生成报告等。                                     |




## 六、$CrossEntropyLoss$
CrossEntropyLoss 是手写数字识别中的核心损失函数。它实际上是 Softmax + Log + NLLLoss 三个函数的集成，下面分别说明这三个函数：
###  $Softmax:$
Softmax回归是一个线性多分类模型，在MINIST手写数字识别问题中，Softmax最终会给出预测值对于10个类别（0~9）出现的概率，最终模型的预测结果就是概率最大的类别。Softmax计算公式如下：

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
其中分子的$z_i$是多分类中的第$i$类的输出值，分母将所有类别的输出值求和，使用指数函数来将其转换为概率，最终将神经网络上一层的原始数据归一化到$[0,1]$,使用指数函数的原因是因为上一层的数据有正有负，所以使用指数函数将其变为大于0的值。具体转换过程如下图所示，可以通过判断哪类的输出概率最大，来判断最后的分类结果。
### $Log:$

  经过Softmax后，还要将其结果取$Log$​(对数),目的是将乘法转化为加法，从而减少计算量，同时保证函数的单调性，因为$ln(x)$单调递增且：
$$
ln(x)+ln(y)=ln(xy)
$$

### $NLLLoss:$
最终使用NLLLoss计算损失，损失函数定义为：
$$
Loss(\hat{Y}, Y) = -Y \log(\hat{Y})
$$
#### 其中的参数含义：
- $\hat{Y}$表示Softmax经过Log​后的值
- $Y$为训练数据对应target的One-hot编码，表示此训练数据对应的target。



## 七、保存训练好的模型
```python
model_save_path = "./mnist_cnn.pth"
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),},model_save_path)
print(f"模型已成功保存到 {model_save_path}")
```

## 八、测试自己的数据
```pyton
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
```
## 训练与测试结果
[![pVSWLY8.png](https://s21.ax1x.com/2025/05/26/pVSWLY8.png)](https://imgse.com/i/pVSWLY8)
### 自定义数据的预测
[![pVSWqFf.png](https://s21.ax1x.com/2025/05/26/pVSWqFf.png)](https://imgse.com/i/pVSWqFf)
#### 可以看到训练后损失已经非常小了，预测的正确率也是非常高。
## 所有代码
```python
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
```
[本文参考文档] https://github.com/IronmanJay/ConvolutionalNeuralNetwork/tree/master/MinistHandWrittenDigitRecognition
