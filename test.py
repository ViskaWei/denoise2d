from src.model import DenoisingModel
import torch
import matplotlib.pyplot as plt
import sys
from utils import get_dataloaders

import matplotlib.pyplot as plt
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
)

def main(args):
    _, val_dl = get_dataloaders(args)
    
    net = DenoisingModel.load_from_checkpoint(args.checkpoint_path)
    
    for batch in val_dl:
        noisy_signal, clean_signal = batch
        
        output, noise_est = net(noisy_signal.cuda())
        outputs = net.loss_and_pme(noisy_signal.cuda(), output, noise_est)
        
        denoised_estimate = outputs['pme_out']
        
        break
    
    fig, axes = plt.subplots(6, 8, sharex=True, sharey=True, figsize=(16, 12))
    
    for i in range(8):
        noisy = noisy_signal[i].cpu()
        img = clean_signal[i]
        denoised_img = torch.clamp(denoised_estimate[i].cpu().detach(), 0, 1)
        
        axes[0, i].imshow(img.permute(1, 2, 0))
        axes[0, i].set_title('Clean')
        axes[0, i].axis('off')
        
        psnr = 10 * torch.log10(1 / torch.mean((img - denoised_img)**2))
        
        axes[1, i].imshow(denoised_img.permute(1, 2, 0))
        axes[1, i].set_title('NN | {psnr:.2f} dB'.format(psnr=psnr))
        axes[1, i].axis('off')
        
        denoised = denoise_tv_chambolle(noisy.permute(1, 2, 0).numpy(), weight=0.1, channel_axis=-1)
        denoised = torch.tensor(denoised).permute(2, 0, 1)
        psnr = 10 * torch.log10(1 / torch.mean((img - denoised)**2))
        axes[2, i].imshow(denoised.permute(1, 2, 0))
        axes[2, i].set_title('TV | {psnr:.2f} dB'.format(psnr=psnr))
        axes[2, i].axis('off')
        
        denoised = denoise_bilateral(noisy.permute(1, 2, 0).numpy(), sigma_color=0.05, sigma_spatial=15, channel_axis=-1)
        denoised = torch.tensor(denoised).permute(2, 0, 1)
        psnr = 10 * torch.log10(1 / torch.mean((img - denoised)**2))
        axes[3, i].imshow(denoised.permute(1, 2, 0))
        axes[3, i].set_title('Bilateral | {psnr:.2f} dB'.format(psnr=psnr))
        axes[3, i].axis('off')
        
        denoised = denoise_wavelet(noisy.permute(1, 2, 0).numpy(), channel_axis=-1, rescale_sigma=True)
        denoised = torch.tensor(denoised).permute(2, 0, 1)
        psnr = 10 * torch.log10(1 / torch.mean((img - denoised)**2))
        axes[4, i].imshow(denoised.permute(1, 2, 0))
        axes[4, i].set_title('Wavelet | {psnr:.2f} dB'.format(psnr=psnr))
        axes[4, i].axis('off')
        
        psnr = 10 * torch.log10(1 / torch.mean((img - noisy)**2))
        
        axes[5, i].imshow(noisy.permute(1, 2, 0))
        axes[5, i].set_title('Noisy | {psnr:.2f} dB'.format(psnr=psnr))
        axes[5, i].axis('off')
        
    fig.tight_layout()
    plt.savefig(f'examples/denoised_{args.dataset}.png', dpi=300)
    
if __name__ == '__main__':
    from utils import get_parser
    parser = get_parser()
    
    args = parser.parse_args()
    
    sys.exit(main(args))