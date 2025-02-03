import torch as th
import lightning as lgt
from src.network import BlindpostModel
from schedulefree import AdamWScheduleFree
from torchvision.transforms.functional import center_crop

class DenoisingModel(lgt.LightningModule):
    def __init__(self, input_channels, fixed_sigma=False, sigma=1.0):
        super().__init__()
        self.num_output_components = input_channels + input_channels * (input_channels + 1) // 2
        self.fixed_sigma = fixed_sigma
        self.sigma = sigma
        
        self.blindspot_net = BlindpostModel(input_channels=input_channels, 
                                               output_channels=self.num_output_components, blindspot=True,
                                               zero_last=False)
        
        if not fixed_sigma:        
            self.noise_estimator = BlindpostModel(input_channels=input_channels, output_channels=1, 
                                              blindspot=False, zero_last=True)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        blindspot = self.blindspot_net(x)
        
        if self.fixed_sigma:
            noise = th.ones(size=(x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
            noise = noise * self.sigma
        else:
            noise = th.nn.functional.softplus(self.noise_estimator(x) - 4.0) + 1e-3
        
        return blindspot, noise
    
    def training_step(self, batch, batch_idx):
        noisy_signal, clean_signal = batch
        
        output, noise_est = self.forward(noisy_signal)
        
        out_dict = self.loss_and_pme(noisy_signal, output, noise_est)
        
        final_loss = out_dict['final_loss']
        mean_regularization = out_dict['mean_reg']
        log_det_loss = out_dict['log_det_loss']
        diff_mean = out_dict['diff_mean']
        mu_x = out_dict['mu_x']
        
        self.log('train_loss', final_loss)
        self.log('train_mean_reg', mean_regularization)
        self.log('train_log_det_loss', log_det_loss)
        self.log('train_diff_mean', diff_mean)
        
        denoised_estimate = out_dict['pme_out']
        psnr = self.mpsnr(denoised_estimate, clean_signal)
        self.log('train_psnr', psnr)
        
        psnr_mu = self.mpsnr(mu_x, clean_signal)
        self.log('train_psnr_mu', psnr_mu)
        
        _, value = self.autocorrelation(denoised_estimate - clean_signal)
        self.log('train_D-C_autocorr', value)
        
        return final_loss
    
    def mpsnr(self, input, target):
        c0 = 20 * th.log10(th.amax(target, dim=[1, 2, 3]) + 1e-6)
        c1 = -10 * th.log10(th.mean((input - target)**2, dim=[1, 2, 3]) + 1e-6)
        
        psnr = c0 + c1
        
        return psnr.mean()
    
    def loss_and_pme(self, noisy_signal, blinspot_output, noise_est):
        """
            Compute the posterior mean estimate of the signal
            and the loss function.
            
            Only the Gaussian noise model is considered!
        """
        batch_size, channels, height, width = noisy_signal.shape

        mu_x = blinspot_output[:, :channels, ...]
        A_c = blinspot_output[:, channels:, ...]
        
        mu_x2 = mu_x.permute(0, 2, 3, 1)
        noisy_signal2 = noisy_signal.permute(0, 2, 3, 1)
        
        upper_triangular = th.tril_indices(channels, channels, 0)
        A = th.zeros(size=(batch_size, channels, channels,
                           height, width), device=noisy_signal.device)
        
        A[:, upper_triangular[0], upper_triangular[1], ...] = A_c
        
        sigma_x = th.einsum('bijkl,bipkl->bjpkl', A, A)
        
        sigma_x = sigma_x.permute(0, 3, 4, 1, 2)
        
        # Eq. (4) from the paper
        I = th.eye(channels, device=noisy_signal.device).view(1, 1, 1, channels, channels)
        sigma_noise = noise_est**2
        sigma_noise = sigma_noise.permute(0, 2, 3, 1).unsqueeze(-1) # BHW11
        sigma_noise = sigma_noise * I
       
        sigma_y = sigma_x + sigma_noise
        sigma_y_inv = th.linalg.inv(sigma_y)
        
        diff = (noisy_signal2 - mu_x2)
        
        results = diff.unsqueeze(-1) * diff.unsqueeze(-2) * sigma_y_inv
        results = results.sum(dim=[-1, -2])
        
        diff = -0.5 * results
        dets = th.linalg.det(sigma_y)
        
        dets = th.maximum(dets, th.tensor(0.0, device=noisy_signal.device))
        
        log_det_loss = 0.5 * th.log(dets)
        
        loss = log_det_loss - diff
        
        reg = - 0.1 * noise_est.mean(dim=1)
        
        loss = loss + reg
        
        final_loss = loss.mean()
        
        # Eq. (6) from the paper
        sigma_x_inv = th.linalg.inv(sigma_x + I * 1e-6)
        sigma_noise_inv = th.linalg.inv(sigma_noise + I * 1e-6)
        
        pme_c1 = th.linalg.inv(sigma_x_inv + sigma_noise_inv + I * 1e-6)
        pme_c2 = sigma_x_inv * mu_x2.unsqueeze(-2)
        pme_c2 = pme_c2.sum(dim=-1)
        
        pme_c2_b = sigma_noise_inv * noisy_signal2.unsqueeze(-2)
        pme_c2 = pme_c2 + pme_c2_b.sum(dim=-1)
        
        pme_out = pme_c1 * pme_c2.unsqueeze(-2)
        pme_out = pme_out.sum(dim=-1)
        pme_out = pme_out.permute(0, 3, 1, 2)
        
        net_std_out = th.maximum(th.linalg.det(sigma_x), th.tensor(0.0, device=noisy_signal.device))**(1.0/6.0)
        noise_std_out = th.maximum(th.linalg.det(sigma_noise), th.tensor(0.0, device=noisy_signal.device))**(1.0/6.0)
        
        return {
            'final_loss': final_loss,
            'pme_out': pme_out,
            'mu_x': mu_x,
            'sigma_x': sigma_x,
            'sigma_noise': sigma_noise,
            'net_std_out': net_std_out,
            'noise_std_out': noise_std_out,
            'mean_reg': reg.mean(),
            'log_det_loss': log_det_loss.mean(),
            'diff_mean': -diff.mean(),
        }
        
    def autocorrelation(self, x):
        f = th.fft.fft2(x, dim=[-2, -1])
        x = th.fft.ifft2(f * th.conj(f), dim=[-2, -1])
        x = th.real(x)
        x = th.fft.fftshift(x, dim=[-2, -1])
        
        # NOTE: get middle rectangle
        center = center_crop(x, (10, 10))
        
        return x, center.sum(dim=[-3, -2, -1]).mean()
        
        
    def validation_step(self, batch, batch_idx):
        noisy_signal, clean_signal = batch
        
        output, noise_est = self.forward(noisy_signal)
        
        out_dict = self.loss_and_pme(noisy_signal, output, noise_est)
        
        final_loss = out_dict['final_loss']
        reg = out_dict['mean_reg']
        log_det_loss = out_dict['log_det_loss']
        diff_mean = out_dict['diff_mean']
        
        self.log('val_loss', final_loss)
        self.log('val_mean_reg', reg)
        self.log('val_log_det_loss', log_det_loss)
        self.log('val_diff_mean', diff_mean)
        
        denoised_estimate = out_dict['pme_out']
        psnr = self.mpsnr(denoised_estimate, clean_signal)
        self.log('val_psnr', psnr)
        
        psnr_mu = self.mpsnr(out_dict['mu_x'], clean_signal)
        self.log('val_psnr_mu', psnr_mu)
        
        _, value = self.autocorrelation(denoised_estimate - clean_signal)
        self.log('val_D-C_autocorr', value)
        
        return final_loss
        
    def configure_optimizers(self):
        print(self.parameters())
        return AdamWScheduleFree(self.parameters(), lr=1e-4)
        
    
if __name__ == '__main__':
    net = DenoisingModel(input_channels=3)
    
    x = th.randn(64, 3, 64, 64)
    x = x - x.min()
    x = x / x.max()
    
    print(x.shape, x.min(), x.max())
    
    net.training_step([x, x], 0)