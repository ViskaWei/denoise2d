from torchvision.transforms import (Compose, ToTensor, RandomHorizontalFlip, 
                                    RandomResizedCrop, RandomRotation, RandomAffine, CenterCrop)
from src.dataset import CIFAR10Noisy, CelebANoisy
from src.hpf import HPFDataset
from torch.utils.data import DataLoader
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to store CIFAR10 dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loader')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--noise_rate', type=float, default=0.1, help='Noise rate')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use')
    parser.add_argument('--checkpoint_path', type=str, default='cifar10-noisy/0i6fyd90/checkpoints/epoch=49-step=19550.ckpt',
                        help='Path to store model checkpoints')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--fp64', action='store_true', help='Use 64-bit floating point precision')
    parser.add_argument('--fixed_sigma', action='store_true', help='Use fixed sigma for noise estimation')
    
    return parser

def get_dataloaders(args):
    if args.dataset == 'cifar10':
        train_transforms = Compose([
            RandomHorizontalFlip(),
            RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
            RandomRotation(15),
            RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
            ToTensor()
        ])
        
        valid_transforms = Compose([
            ToTensor()
        ])
        train_ds = CIFAR10Noisy(root=args.data_dir, train=True, download=True, transform=train_transforms)
        val_ds = CIFAR10Noisy(root=args.data_dir, train=False, download=True, transform=valid_transforms)
    
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'celeba':
        train_transforms = Compose([
            RandomHorizontalFlip(),
            RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
            RandomRotation(15),
            RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
            CenterCrop(128),
            ToTensor()
        ])
        
        valid_transforms = Compose([
            CenterCrop(128),
            ToTensor()
        ])
        
        train_ds = CelebANoisy(root=args.data_dir, split='train', download=True, transform=train_transforms)
        val_ds = CelebANoisy(root=args.data_dir, split='valid', download=True, transform=valid_transforms)
        
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'hpf':
        train_ds = HPFDataset(root_dir=args.data_dir, crop_size=64, mode='train', transform=True)
        val_ds = HPFDataset(root_dir=args.data_dir, crop_size=64, mode='val', transform=False)
    
    
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dl, val_dl