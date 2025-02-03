import torch as th
import torch.nn as nn
import numpy as np

class BlindpostModel(nn.Module):
    def __init__(self, input_channels, output_channels, blindspot, zero_last):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blindspot = blindspot
        self.zero_last = zero_last
        
        super(BlindpostModel, self).__init__()
        
        # NOTE: Initialize the model with a dummy input to create the parameters.
        x = th.randn(1, input_channels, 32, 32)
        _ = self.forward(x)

    def forward(self, x):
        bs, *_ = x.shape
        x = th.cat([self.rotate(x, a) for a in [0, 90, 180, 270]], dim=0)
      
        pool0 = x
        x = self.LR(self.conv(x, 'enc_conv0', 48, blindspot=self.blindspot))
        x = self.LR(self.conv(x, 'enc_conv1', 48, blindspot=self.blindspot))
        x = self.down(x, 'pool1', blindspot=self.blindspot); pool1 = x
        
        x = self.LR(self.conv(x, 'enc_conv2', 48, blindspot=self.blindspot))
        x = self.down(x, 'pool2', blindspot=self.blindspot); pool2 = x
        
        x = self.LR(self.conv(x, 'enc_conv3', 48, blindspot=self.blindspot))
        x = self.down(x, 'pool3', blindspot=self.blindspot); pool3 = x
        
        x = self.LR(self.conv(x, 'enc_conv4', 48, blindspot=self.blindspot))
        x = self.down(x, 'pool4', blindspot=self.blindspot); pool4 = x

        x = self.LR(self.conv(x, 'enc_conv5', 48, blindspot=self.blindspot))
        x = self.down(x, 'pool5', blindspot=self.blindspot)
        
        x = self.LR(self.conv(x, 'enc_conv6', 48, blindspot=self.blindspot))

        x = self.up(x, 'upsample5')
        x = self.concat('concat5', [x, pool4])
        x = self.LR(self.conv(x, 'dec_conv5a', 96, blindspot=self.blindspot))
        x = self.LR(self.conv(x, 'dec_conv5b', 96, blindspot=self.blindspot))
        
        x = self.up(x, 'upsample4')
        x = self.concat('concat4', [x, pool3])
        x = self.LR(self.conv(x, 'dec_conv4a', 96, blindspot=self.blindspot))
        x = self.LR(self.conv(x, 'dec_conv4b', 96, blindspot=self.blindspot))
    
        x = self.up(x, 'upsample3')
        x = self.concat('concat3', [x, pool2])
        x = self.LR(self.conv(x, 'dec_conv3a', 96, blindspot=self.blindspot))
        x = self.LR(self.conv(x, 'dec_conv3b', 96, blindspot=self.blindspot))
        
        x = self.up(x, 'upsample2')
        x = self.concat('concat2', [x, pool1])
        x = self.LR(self.conv(x, 'dec_conv2a', 96, blindspot=self.blindspot))
        x = self.LR(self.conv(x, 'dec_conv2b', 96, blindspot=self.blindspot))
        
        x = self.up(x, 'upsample1')
        x = self.concat('concat1', [x, pool0])
        
        x = self.LR(self.conv(x, 'dec_conv1a', 96, blindspot=self.blindspot))
        x = self.LR(self.conv(x, 'dec_conv1b', 96, blindspot=self.blindspot))
        x = th.nn.functional.pad(x[:, :, :-1, :], (0, 0, 1, 0))
        x = th.split(x, [bs, bs, bs, bs], dim=0)
        x = [self.rotate(y, a) for y, a in zip(x, [0, 270, 180, 90])]
        x = th.cat(x, dim=1)
        
        x = self.LR(self.conv(x, 'nin_a', 96*4, size=1, blindspot=self.blindspot))
        x = self.LR(self.conv(x, 'nin_b', 96, size=1, blindspot=self.blindspot))
        x = self.conv(x, 'nin_c', self.output_channels, size=1, gain=1.0, zero_weights=self.zero_last, blindspot=self.blindspot)
        
        return x
    
    def conv(self, x, name, out_channels, size=3, gain=np.sqrt(2), zero_weights=False, blindspot=False):
        if blindspot: assert (size % 2) == 1
        ofs = 0 if (not blindspot) else size // 2
        
        _, in_channels, _, _ = x.shape

        kernel_shape = [out_channels, in_channels, size, size]
        kernel_std = gain / np.sqrt(np.prod(kernel_shape[1:])) # He init.
        
        if not hasattr(self, name + '_kernel'):
            kernel = th.nn.Parameter(th.zeros(*kernel_shape) if zero_weights else th.randn(*kernel_shape) * kernel_std)
            self.register_parameter(name + '_kernel', kernel)
        else:
            kernel = getattr(self, name + '_kernel')
        
        if not hasattr(self, name + '_bias'):
            b = th.nn.Parameter(th.zeros(out_channels))
            self.register_parameter(name + '_bias', b)
        else:
            b = getattr(self, name + '_bias')
        
        if ofs > 0:
            x = th.nn.functional.pad(x, (0, 0, ofs, 0))
        
        x = th.nn.functional.conv2d(input=x, weight=kernel, bias=None, stride=1, padding='same') + b.view(1, -1, 1, 1)
        
        if ofs > 0:
            x = x[:, :, :-ofs, :]

        return x
    
    def up(self, x, name):
        batch, channels, height, width = x.shape
        
        # NOTE: duplicating pixel values in a 2x2 grid for each pixel
        x = x.view(batch, channels, height, 1, width, 1)
        x = x.repeat(1, 1, 1, 2, 1, 2)
        x = x.view(batch, channels, height * 2, width * 2)
        
        return x
    
    def down(self, x, name, blindspot=False):
        if blindspot:
            x = th.nn.functional.pad(x[:, :, :-1, :], (0, 0, 1, 0))
        
        x = th.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        
        return x
    
    def concat(self, name, layers):
        return th.cat(layers, dim=1)

    def rotate(self, x, angle):
        if angle == 0: return x
        elif angle == 90: return x.flip(dims=(-1,)).permute(0, 1, 3, 2)
        elif angle == 180: return x.flip(dims=(-1, -2,))
        elif angle == 270: return x.flip(dims=(-2,)).permute(0, 1, 3, 2)

    def LR(self, n, alpha=0.1):
        return th.nn.functional.leaky_relu(n, negative_slope=alpha)

if __name__ == '__main__':
    model = BlindpostModel(3, 3, True, True)
    x = th.randn(1, 3, 32, 32)
    print([name for name, param in list(model.named_parameters())])
    y = model(x)
    print([name for name, param in list(model.named_parameters())])
    print(y.shape)