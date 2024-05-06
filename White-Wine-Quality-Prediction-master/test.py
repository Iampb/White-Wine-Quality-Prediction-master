import torch 
import os

from matplotlib import pyplot as plt

from Network.Net import Net
from dataloader import get_x_y

X, y = get_x_y('winequality-red.csv')
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long().add(-3)
lists = os.listdir("./output2")
lists = lists[:-1]
lists.sort(key=lambda x:int(x.split('epoch_')[1].split('.pt')[0]))
epoch_list = []
acc_list = []
i = 1
for fileName in lists:
    param_dic = torch.load(f'./output2/{fileName}')
    net = Net()
    net.load_state_dict(param_dic)

    if torch.cuda.is_available(): # 使用GPU，将数据载入cuda
        X, y = X.cuda(), y.cuda()
        net.cuda()

    net.eval()
    out = net(X)
    pred = torch.max(out, 1)[1]
    print(pred)
    # print(y)
    num_correct = (pred == y).sum()
    print(f'Model: {fileName}, Acc: {num_correct.item() / (len(y)) * 100:.2f}%')
    acc_list.append((num_correct.item() / (len(y)) * 100))
    epoch_list.append(i)
    i += 1
#plt.scatter(epoch_list,acc_list)
plt.plot(acc_list)
plt.xlabel("epoch")
plt.ylabel("acc")
plt.show()