import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.utils.data as data

batch = 1000
class RnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer = nn.LSTM(28,64,1,batch_first=True)
        self.output_layer = nn.Linear(64,10)
    def forward(self, x):
        #NCHW-->NS(C*H)V(W)
        input = x.view(-1,28,28)

        # h0 = torch.zeros(1,batch,64)
        # c0 = torch.zeros(1, batch, 64)

        outputs,(hn,cn) = self.rnn_layer(input)
        # print(outputs.size())
        output = outputs[:,-1,:]#只要NSV的最后一个S的数据
        output = self.output_layer(output)
        # print(output.size())
        return output

if __name__ == '__main__':
    train_dataset = dataset.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    train_dataLoader = data.DataLoader(train_dataset,batch_size=batch,shuffle=True)
    test_dataset = dataset.MNIST(
        root="data",
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )
    test_dataLoader = data.DataLoader(test_dataset, batch_size=batch, shuffle=True)

    net = RnnNet()
    opt = torch.optim.Adam(net.parameters())

    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(100000):
        for i ,(x,y) in enumerate(train_dataLoader):
            output = net(x)
            loss = loss_fun(output,y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0:
                # print("loss",loss.item())
                for xs,ys in test_dataLoader:
                    out = net(xs)
                    test_out = torch.argmax(out,dim=1)
                    acc = torch.mean(torch.eq(test_out,ys).float()).item()
                    # print("acc",acc)
                    break
