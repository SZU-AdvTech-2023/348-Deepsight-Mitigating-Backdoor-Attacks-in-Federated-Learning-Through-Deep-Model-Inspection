import torch
from models.Net import CNNMnist
from models.Test import test_bd
from models.Update import Dataset_flips
from utils.options import args_parser
from models.MnistBackdoor import Mnist_bd

args = args_parser()
args.device = torch.device('cuda:0')
net = CNNMnist().to(args.device)

net.load_state_dict(torch.load('./save_model/model_fa_bd_{}.pth'.format(19)))
# net.load_state_dict(torch.load('./saved_updates/update_-2.pth'))

datatest = Dataset_flips(task=args.dataset, train=True)
d = Mnist_bd(train=False)
test_bd(net, d, args)
# for images, labels in d:
#     img_arr = np.array(images[0])
#     plt.imshow(img_arr)
#     plt.show()
#     print(labels)
