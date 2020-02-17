import torch
from torch.utils.data import DataLoader
import torch.nn as nn # 신경망은 torch.nn 패키지를 사용하여 생성
from dataset import LoadDataset
import os

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
    
    # Hyperparameters.
    N_EPOCHS = 2000

    # Data setup.
    dataset = LoadDataset(root_dir="data/")
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

    # Loss function & optimizer setup.
    criterion = torch.nn.MSELoss()
    optimizer = saved_optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
    

    # Train.
    min_loss = saved_loss
    if min_loss is None:
        min_loss = float("inf")

    for epoch in range(N_EPOCHS):
        for i_batch, sample_batched in enumerate(dataloader):
            watermarked, original = sample_batched
            #print(watermarked.shape, original.shape)
            original = model.resize(sample=original) 
     
            output = model(x=watermarked)
            loss = criterion(output, original)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # TODO: Move this into `debug()` method.
        if epoch == 0 or epoch % 100 == 0:
            print(">> epoch # {}: {}".format(epoch, loss.data))
            
            if loss < min_loss:
                print(">> updating weights.")
                model.save(optimizer=optimizer, loss=loss)
                min_loss = loss
