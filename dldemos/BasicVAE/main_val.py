
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from VAE_model import AE,VAE, save_recon_images, vae_loss, bce_kld_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np 


if __name__ == "__main__":





    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='../dl_dataset/MNIST_dataset', train=True, download=True, transform=transform)

    # 划分训练集和测试集（按文件划分）
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    x_test, _ = next(iter(test_loader))
    x_test = x_test.to(device)




    MODEL_NAME ="VAE"
    KL_RATIO = 0.1



    match MODEL_NAME :
        case 'AE' :
            model = AE()
        case 'VAE' :
            model = VAE()
        case _ :
            print('无效模型')

    model = model.to(device)

    # 初始化模型
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fn = nn.MSELoss()


    # 初始化记录器
    total_loss = []

    # 训练 VAE
    for epoch in range(20):
        model.train()
        current_train_loss = []

        for x, _ in train_loader:
            x = x.to(device)
            model.zero_grad()
            recon, mu, logvar = model(x)
            bce_loss, kld_loss = bce_kld_loss(recon, x, mu, logvar)

            loss = bce_loss + kld_loss

            loss.backward()
            optimizer.step()
        
            # 记录每个epoch的bce_loss和kld_loss
            current_train_loss.append([bce_loss.item(), kld_loss.item(), loss.item()])
        mean_current_train_loss = [np.array(current_train_loss)[:,0].mean(), np.array(current_train_loss)[:,1].mean(), np.array(current_train_loss)[:,2].mean()   ]

        
        print(f"[VAE] Epoch {epoch+1}, Loss: {loss.item():.4f}")


        model.eval()
        with torch.no_grad():
            current_test_loss = []
            for x, _ in test_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                bce_loss, kld_loss = bce_kld_loss(recon, x, mu, logvar)

                loss = bce_loss + kld_loss

                # # 记录每个epoch的bce_loss和kld_loss
                # bce_losses.append(bce_loss.item())
                # kld_losses.append(kld_loss.item())
                current_test_loss.append([bce_loss.item(), kld_loss.item(), loss.item()])
            mean_current_test_loss = [np.array(current_test_loss)[:,0].mean(), np.array(current_test_loss)[:,1].mean(), np.array(current_test_loss)[:,2].mean()   ]
        # total_loss.append(mean_current_train_loss, mean_current_test_loss)
        total_loss.append([mean_current_train_loss[0], mean_current_train_loss[1], mean_current_train_loss[2], mean_current_test_loss[0], mean_current_test_loss[1], mean_current_test_loss[2]])



    # # 绘制bce_loss和kld_loss的曲线
    # plt.figure()
    # plt.plot(bce_losses, label='BCE Loss')
    # plt.plot(kld_losses, label='KLD Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('BCE and KLD Loss Curves')
    # plt.legend()

    # plt.show()
    # # plt.savefig('./output/loss_curves.png')
    # # plt.close()


