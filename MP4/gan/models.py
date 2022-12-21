import torch
import torch.nn as nn
from .spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        
        # Hint: Apply spectral normalization to convolutional layers. 
        #       Input to SpectralNorm should be your conv nn module
        self.conv1 = SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, padding=1))
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = SpectralNorm(nn.Conv2d(512, 1024, 4, stride=2, padding=1))
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = SpectralNorm(nn.Conv2d(1024, 1, 4, stride=1))
    
    def forward(self, x):
        # maybe leaky relu at the end?
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.bn2(x)
        x = self.lrelu(self.conv3(x))
        x = self.bn3(x)
        x = self.lrelu(self.conv4(x))
        x = self.bn4(x)
        x = self.conv5(x)
        return x
    
class Discriminator2(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator2, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        
        self.conv1 = nn.Conv2d(3, 128, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, 4, stride=1)
    
    def forward(self, x):
        # maybe leaky relu at the end?
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.bn2(x)
        x = self.lrelu(self.conv3(x))
        x = self.bn3(x)
        x = self.lrelu(self.conv4(x))
        x = self.bn4(x)
        x = self.conv5(x)
        return x

class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # maybe padding all layers or only layers 2-5 or none?
        
        self.conv1 = nn.ConvTranspose2d(noise_dim, 1024, 4, stride=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
    
    def forward(self, x):
        x = x.view(x.shape[0], self.noise_dim, 1, 1)
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.tanh(x)
        return x
    

