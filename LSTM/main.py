import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from collections import Counter
import pandas as pd
import numpy as np

# 数据

df = pd.read_csv('surnames.csv')
df.columns=['Name','Language']

# 将国家标签转换为数值
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Language'])
num_classes = len(label_encoder.classes_)
torch.save(label_encoder, 'label_encoder.pth')
# 字符级别的Tokenizer（这里使用简单的字符索引映射）
char_to_idx = {char: idx for idx, char in enumerate(sorted(set(''.join(df['Name']))))}
print(char_to_idx)
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# 将姓名转换为索引序列
def name_to_indices(name):
    return [char_to_idx[char] for char in name]

df['Indices'] = df['Name'].apply(name_to_indices)
# 固定序列长度（选择最长姓名的长度，为了演示简单，这里不额外加padding）
max_length = max(len(indices) for indices in df['Indices'])
print(df['Indices'])
# 创建PyTorch Dataset
class NameDataset(Dataset):
    def __init__(self, indices, labels):
        self.indices = indices
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 将索引序列填充到max_length
        seq = self.indices[idx] + [0] * (self.max_length - len(self.indices[idx]))
        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

dataset = NameDataset(df['Indices'].tolist(), df['Label'].tolist())

# 拆分为训练集和测试集
train_size = int(0.7 * len(dataset))
test_size = int(0.2 *len(dataset))
valid_size = len(dataset) - train_size - test_size
train_dataset, test_dataset,valid_dataset = random_split(dataset, [0.7, 0.2, 0.1])
# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
valid_loader = DataLoader(valid_dataset,batch_size=8, shuffle=False)
# 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out

# 模型参数
vocab_size = len(char_to_idx) + 1  # +1 for padding
embedding_dim = 32
hidden_dim = 64
num_layers = 1

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters())
#optimizer = optim.Adagrad(model.parameters())
Accuracy_list_test = []
loss_train = []
# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    loss_train.append(round(loss.item(),4))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    Accuracy_list_test.append(100 * correct / total)
torch.save(model, f'model_epoch_{num_epochs}_numlays{num_layers}_Adam.pth')
print(Accuracy_list_test)
print(loss_train)
# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in valid_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy of the model on the valid data: {100 * correct / total:.2f}%')
print()
# 预测
def predict_name(model, name):
    indices = name_to_indices(name) + [0] * (max_length - len(name))
    indices = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(indices)
        _, predicted = torch.max(output.data, 1)
    return label_encoder.inverse_transform([predicted.item()])[0]

test_name = 'Yang'
predicted_language = predict_name(model, test_name)
print(f'The predicted language for "{test_name}" is: {predicted_language}')
