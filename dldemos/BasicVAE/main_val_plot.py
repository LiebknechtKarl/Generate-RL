
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
    dataset = torchvision.datasets.MNIST(root='../../dl_dataset/MNIST_dataset', train=True, download=True, transform=transform)

    # 划分训练集和测试集（按文件划分）
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    x_test, _ = next(iter(test_loader))
    x_test = x_test.to(device)




    MODEL_NAME ="VAE"



    match MODEL_NAME :
        case 'AE' :
            model = AE()
        case 'VAE' :
            model = VAE()
        case _ :
            print('无效模型')

    model = model.to(device)


    ratio_list = [10, 1 ,1e-1, 1e-2, 1e-3, 0 ]

    total_loss_dic = {}
    for ratio_step  in range(len(ratio_list)) :
        KL_RATIO = ratio_list[ratio_step]

        print('比例', KL_RATIO)
        # 初始化模型
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        loss_fn = nn.MSELoss()
        # 初始化记录器
        total_loss = []

        # 训练 VAE
        for epoch in range(100):
            model.train()
            current_train_loss = []

            for x, _ in train_loader:
                x = x.to(device)
                model.zero_grad()
                recon, mu, logvar = model(x)
                bce_loss, kld_loss = bce_kld_loss(recon, x, mu, logvar)

                # loss = bce_loss + kld_loss
                loss = bce_loss + KL_RATIO*kld_loss

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

            total_loss_dic[str(ratio_step)] = total_loss

    print(total_loss_dic)


    import pickle
    # 保存字典到 pickle 文件
    with open('dldemos/BasicVAE/total_loss_dic.pkl', 'wb') as f:
        pickle.dump(total_loss_dic, f)


    import json
    # 保存字典到 JSON 文件
    with open('dldemos/BasicVAE/total_loss_dic.json', 'w') as f:
        json.dump(total_loss_dic, f)





    import json
    import numpy as np

    # 从 JSON 文件加载字典
    with open('dldemos/BasicVAE/total_loss_dic.json', 'r') as f:
        total_loss_dic = json.load(f)

    print(total_loss_dic['0'])

    ratio_list = [10, 1 ,1e-1, 1e-2, 1e-3, 0 ]

    import matplotlib.pyplot as plt


    # loss_0 = total_loss_dic['0']
    loss_0 = np.array(total_loss_dic['0'])[:,3]
    loss_1 = np.array(total_loss_dic['1'])[:,3]
    loss_2 = np.array(total_loss_dic['2'])[:,3]
    loss_3 = np.array(total_loss_dic['3'])[:,3]
    loss_4 = np.array(total_loss_dic['4'])[:,3]
    loss_5 = np.array(total_loss_dic['5'])[:,3]

    # 创建图形和坐标轴
    plt.figure()
    # 绘制两条折线
    # ratio_list = [10, 1 ,1e-1, 1e-2, 1e-3, 0 ]

    plt.plot(loss_0, label='Line 10')
    plt.plot(loss_1, label='Line 1')
    plt.plot(loss_2, label='Line 1e-1')
    plt.plot(loss_3, label='Line 1e-2')
    plt.plot(loss_4, label='Line 1e-3')
    plt.plot(loss_5, label='Line 0')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()