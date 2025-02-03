from torchvision.datasets import CIFAR10, CelebA
import torch

class CIFAR10Noisy(CIFAR10):
    def __init__(self, root, noise_type='gaussian', noise_rate=0.1, **kwargs):
        super().__init__(root, **kwargs)
        self.noise_type = noise_type
        self.noise_rate = noise_rate

    def _add_noise(self, img):
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(img) * self.noise_rate # zero mean, additive noise
            img = img + noise
        else:
            raise ValueError(f'Noise type {self.noise_type} not supported.')
        return torch.clamp(img, 0, 1)

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        
        noisy_img = self._add_noise(img)
        
        return noisy_img, img

    def __len__(self):
        return super().__len__()
    
class CelebANoisy(CelebA):
    def __init__(self, root, noise_type='gaussian', noise_rate=0.1, **kwargs):
        super().__init__(root, **kwargs)
        self.noise_type = noise_type
        self.noise_rate = noise_rate

    def _add_noise(self, img):
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(img) * self.noise_rate # zero mean, additive noise
            img = img + noise
        else:
            raise ValueError(f'Noise type {self.noise_type} not supported.')
        return torch.clamp(img, 0, 1)

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        
        noisy_img = self._add_noise(img)
        
        return noisy_img, img

    def __len__(self):
        return super().__len__()

if __name__ == '__main__':
    from torchvision.transforms import ToTensor
    
    transform = ToTensor()
    
    ds = CIFAR10Noisy(root='data', noise_type='gaussian', noise_rate=0.1,
                      train=True, download=True, transform=transform)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 8, sharex=True, sharey=True, figsize=(16, 4))
    
    for i in range(8):
        noisy_img, img = ds[i]
        axes[0, i].imshow(img.permute(1, 2, 0))
        axes[0, i].set_title('Clean')
        axes[0, i].axis('off')
        
        psnr = 10 * torch.log10(1 / torch.mean((img - noisy_img)**2))
        
        axes[1, i].imshow(noisy_img.permute(1, 2, 0))
        axes[1, i].set_title('Noisy | {psnr:.2f} dB'.format(psnr=psnr))
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('examples/data_noisy_cifar10.png', dpi=300)
    
    ds = CelebANoisy(root='data', noise_type='gaussian', noise_rate=0.1,
                        split='train', download=True, transform=transform)
    
    fig, axes = plt.subplots(2, 8, sharex=True, sharey=True, figsize=(16, 4))
    
    for i in range(8):
        noisy_img, img = ds[i]
        axes[0, i].imshow(img.permute(1, 2, 0))
        axes[0, i].set_title('Clean')
        axes[0, i].axis('off')
        
        psnr = 10 * torch.log10(1 / torch.mean((img - noisy_img)**2))
        
        axes[1, i].imshow(noisy_img.permute(1, 2, 0))
        axes[1, i].set_title('Noisy | {psnr:.2f} dB'.format(psnr=psnr))
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('examples/data_noisy_celeba.png', dpi=300)