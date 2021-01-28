import PIL
import torch

from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

def load_params():
    np.set_printoptions(suppress=True)
    path = "./data/"
    model = EfficientNet.from_pretrained("efficientnet-b0")
    module_list = ""
    
    for m in model.state_dict():
        #メタデータの読み込み
        size = model.state_dict()[m].size()
        data = str(len(size))
        for s in size:
            data += "," + str(s)
        
        #データの読み込み
        weight = model.state_dict()[m].view(-1).numpy()
        for w in weight:
            data += "," + str(w)
        
        #書き出し
        module_list += (m + "\n");
        with open(path + m + ".txt", 'w') as f:
            print(data, file=f)
        
    
    #with open(path + "module_list.txt", 'w') as f:
    #    print(module_list, file=f)

def main():
    #load_params()
    model = EfficientNet.from_pretrained("efficientnet-b0")
    image_size = 224
    val_dir = 'val'
    
    val_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    val_set = datasets.ImageFolder(val_dir, val_transforms)
    val_loader = torch.utils.data.DataLoader(val_set)
    
    model.eval()
    
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            output = model(images)
            
    
if __name__ == "__main__":
    main()
    








































