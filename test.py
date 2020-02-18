import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
from dataset import LoadDataset
class Autoencoder(nn.Module):
    '''
    참고
    https://github.com/tallosan/DeWatermarker/tree/master/data
    https://tutorials.pytorch.kr/beginner/blitz/neural_networks_tutorial.html
    nn 은 모델을 정의하고 미분하는데 autograd 를 사용하는데, nn.Module 은 계층(layer)과 output 을 반환하는 forward(input) 메서드를 포함한다.
    '''
    FPATH = "saved_model.pt"

    # documentation images shape = 1 x 2306 x 1728
    def __init__(self, inpt_shape):
        super().__init__()
        
        self.encoder = nn.Sequential(
            #input = batch x 1 x 2306 x 1728
            #output = batch x 16 x 1152 x 863
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(True),

            #input = batch x 1 x 1152 x 863
            #output = batch x 32 x 575 x 430
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3
            ),
            nn.ReLU(inplace=True),

            #input = batch x 32 x 575 x 430
            #output = batch x 64 x 286 x 214
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            ),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, padding=0)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3
            ),
            nn.ReLU(inplace=True)
        )
    
    
    def forward(self, x):
        """
        Perform the forward pass on the given input.
        Args:
            x (Tensor): The input to perform the forward pass on.
        """
        x = self.resize(x)
        encoded_x = self.encoder(x)
        pool_x, indices = self.pool(encoded_x)
        unpool_x = self.unpool(pool_x, indices)
        decoded_x = self.decoder(unpool_x)

        return decoded_x

    def load(self):
        """
        Load in any existing weights belonging to this model.
        """
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        loss = None
        try:
            checkpoint = torch.load(self.FPATH)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
                
            if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if isinstance(checkpoint, dict) and 'loss' in checkpoint:    
                loss = checkpoint['loss']

            self.eval()
            return optimizer, loss
            
        except FileNotFoundError:
            msg = "No existing model to initialize from. Creating new one ..."
            print(msg)
            return None, None

    def save(self, optimizer, loss):
        """
        Save the current state of this model.
        """
        torch.save({
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, self.FPATH)

    def resize(self, sample):
        """
        Resize a sample so that it can be inputted to a PyTorch
        Conv2D layer. We need to do this, as PyTorch our input
        is expected in the following shape:
            (batch_size, n_channels, height, width)
        """
        
        #print(sample.shape) # torch.Size([1, 2306, 1728, 3]) -> torch.Size([1, 2306, 1728])
        sample = sample.reshape(1,2306,1728,1)
        return sample.permute(0, 3, 1, 2).type("torch.FloatTensor")



if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    # Data setup.
    dataset = LoadDataset(root_dir='data_test/')
    INPT_SHAPE = dataset.len
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=3
    )
    
    # Model setup. Note, we have the option to load in an existing model.
    model = Autoencoder(inpt_shape=INPT_SHAPE)
    saved_optimizer, saved_loss = model.load()
    
    for i_batch, sample_batched in enumerate(dataloader):
        watermarked, original = sample_batched
        output = model(x=watermarked).detach().numpy()
        output_pic = output.reshape(2306, 1728, 1).astype(int) #tiff float형으로 저장이 안돼서, uint8로 하면 노이즈 안없어짐
        
        cv2.imwrite('C:/Users/soyeon/Desktop/Autoencoder_watermark_remove/output{}.tiff'.format(int(i_batch)), output_pic)
   
    print('Finished!')
