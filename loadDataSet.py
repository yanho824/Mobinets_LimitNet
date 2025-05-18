from torchvision import datasets, transforms

# 加载 CIFAR-100 训练集
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True)

# 加载 CIFAR-100 测试集
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)