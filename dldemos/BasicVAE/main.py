
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

    # 初始化模型
    ae = AE().to(device)
    vae = VAE().to(device)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-4)
    opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()




    # # 训练 AE
    # for epoch in range(5):
    #     ae.train()
    #     for x, _ in train_loader:
    #         x = x.to(device)
    #         opt_ae.zero_grad()
    #         x_hat = ae(x)
    #         loss = loss_fn(x_hat, x)
    #         loss.backward()
    #         opt_ae.step()
    #     print(f"[AE] Epoch {epoch+1}, Loss: {loss.item():.4f}")
    #     ae.eval()
    #     with torch.no_grad():
    #         recon = ae(x_test).cpu()
    #         save_recon_images(recon, epoch, out_dir='./output/ae')
    # ############################------------------------------------------------

    # # 训练 VAE
    # for epoch in range(5):
    #     vae.train()
    #     for x, _ in train_loader:
    #         x = x.to(device)
    #         opt_vae.zero_grad()
    #         recon, mu, logvar = vae(x)
    #         # loss = vae_loss(recon, x, mu, logvar)
    #         bce_loss, kld_loss = bce_kld_loss(recon, x, mu, logvar)
    #         loss = bce_loss + kld_loss
    #         loss.backward()
    #         opt_vae.step()
    #     print(f"[VAE] Epoch {epoch+1}, Loss: {loss.item():.4f}")
    #     vae.eval()
    #     with torch.no_grad():
    #         recon, _, _ = vae(x_test)
    #         save_recon_images(recon.cpu(), epoch, out_dir='./output/vae')

    # ############################------------------------------------------------

    # 初始化记录器
    bce_losses = []
    kld_losses = []

    # 训练 VAE
    for epoch in range(5):
        vae.train()
        for x, _ in train_loader:
            x = x.to(device)
            opt_vae.zero_grad()
            recon, mu, logvar = vae(x)
            bce_loss, kld_loss = bce_kld_loss(recon, x, mu, logvar)
            loss = bce_loss + kld_loss
            loss.backward()
            opt_vae.step()
        
            # 记录每个epoch的bce_loss和kld_loss
            bce_losses.append(bce_loss.item())
            kld_losses.append(kld_loss.item())
        
        print(f"[VAE] Epoch {epoch+1}, Loss: {loss.item():.4f}")
        vae.eval()
        with torch.no_grad():
            recon, _, _ = vae(x_test)
            save_recon_images(recon.cpu(), epoch, out_dir='./output/vae')

    # 绘制bce_loss和kld_loss的曲线
    plt.figure()
    plt.plot(bce_losses, label='BCE Loss')
    plt.plot(kld_losses, label='KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BCE and KLD Loss Curves')
    plt.legend()

    plt.show()
    # plt.savefig('./output/loss_curves.png')
    # plt.close()


