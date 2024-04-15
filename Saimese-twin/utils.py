import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import  Dataset, DataLoader
import matplotlib.pyplot as plt
import glob
import torch 


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, highdose_path, lowdose_path , size=None):
        self.size = size
        
        # self.images = [os.path.join(path, file) for file in os.listdir(path)]
        
        self.highDoseimages = sorted_list(highdose_path)[:100]
        self.lowDoseimage = sorted_list(highdose_path)[:100]
        self._length = len(self.lowDoseimage)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.load(image_path).astype(np.uint8)
        image = image.transpose(1,2,0)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image  = torch.from_numpy(image)
        return image

    
    def __getitem__(self, i):
      
        highDose = self.preprocess_image(self.highDoseimages[i])
        lowDose = self.preprocess_image(self.lowDoseimage[i])
        return highDose , lowDose




def sorted_list(path): 
    
    """ function for getting list of files or directories. """
    
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list
    
    return tmplist



def load_data(args):
    
    # train_data = ImagePaths(args.dataset_path, size=512)
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    dataset = ImagePaths(args.fulldose_dataset_path + "/*" , args.lowdose_dataset_path + '/*', size= args.image_size)
    # low_dose_data = ImagePaths(args.lowdose_dataset_path + "/*" , size= args.image_size)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # low_dose_loader  =  DataLoader(low_dose_data, batch_size=args.batch_size, shuffle=False)
    return data_loader 


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
