import utils
from utils import load_data
import argparse

parser = argparse.ArgumentParser(description="VQGAN")
args = dict()

parser.add_argument('--fulldose-dataset-path', type=str, default='../Mayo_data/train/full_1mm', help='Path to data (default: /CT_Data_imgs)')
parser.add_argument('--lowdose-dataset-path', type=str, default='../Mayo_data/train/quarter_1mm', help='Path to data (default: /data)')
parser.add_argument('--image-size', type=int, default=512, help='Image height and width (default: 256)')
parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')

args = parser.parse_args()

train_loader , train_dataset = load_data(args)
img1, img2 = train_dataset.__getitem__(0)
for i,j in zip(enumerate(range(0,10)) , train_loader):
    print(i)
    import ipdb
    ipdb.set_trace()