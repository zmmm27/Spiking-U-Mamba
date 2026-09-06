The article link is:https://www.mdpi.com/2079-9292/15/17/3879 
Our code has been uploaded.The configured environment is:
cuda-nvcc                 11.8.89                      
cudatoolkit               11.8.0               
cupy-cuda11x              13.6.0 
causal-conv1d             1.4.0
mamba-ssm                 2.2.2
spikingjelly              0.0.0.0.14
torch                     2.1.1+cu118              
torchaudio                2.1.1+cu118              
torchinfo                 1.8.0                    
torchvision               0.16.1+cu118

The directory format of the CAMUS dataset is：
CAMUS/
├── train/
│   ├── images/  
         ├──patient0001_2CH_ED_1096_aug000.png
         ├──patient0001_2CH_ED_1096_aug001.png
          ……        
│   └── masks/     
         ├──patient0001_2CH_ED_1096_aug000.png
         ├──patient0001_2CH_ED_1096_aug001.png
          ……                 
├── valid/
│   ├── images/   
         ├──patient0023_2CH_ED_0136.png  
          …… 
│   └── masks/  
         ├──patient0023_2CH_ED_0136.png
         ……
├── test/

         
The directory format of the EchoNet-Dynamic dataset is:
EchoNet-Dynamic/
├── TRAIN/
│   ├── images/
│   │   ├── {FileName}_ED_{FrameNum}.png
│   │   └── {FileName}_ES_{FrameNum}.png
│   └── masks/
│       ├── {FileName}_ED_{FrameNum}.png
│       └── {FileName}_ES_{FrameNum}.png
├── VAL/
│   └── ...
├── TEST/

