import os
import os.path as op
import argparse
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import *
from linear import LinearModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(LinearModel):
    def __init__(self, n_classes):
        super(Model, self).__init__(n_classes)
    
    def forward(self, x, mode):
        if mode == 'knn':
            return self.backbone(x) 
        elif mode == 'linear':
            return self.classifier[0](self.backbone(x))


def plot_tsne(model, data_loader, mode, savepath):
    model.eval()
    X, y = [], []
    with torch.no_grad():
        for batch in data_loader:
            data, target = batch['img'], batch['label']
            feature = model(data.to(device), mode)
            # print(feature.shape)
            X.append(feature)
            y.append(target.to(device).long())
        X = torch.cat(X, dim=0).contiguous() # [N, D]
        y = torch.cat(y, dim=0).contiguous() # [N,]
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500, random_state=16)
    tsne_results = tsne.fit_transform(X)
    colors = plt.cm.get_cmap('viridis', len(np.unique(y)))
    plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y, cmap=colors) 
    plt.savefig(savepath, dpi=300)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/soumitri/projects/def-josedolz/soumitri/misc/SmallSSL/data')
    parser.add_argument('--out_root', type=str, default='/home/soumitri/projects/def-josedolz/soumitri/misc/SmallSSL/outputs')
    parser.add_argument('--load_model', type=str, help='path to pretrained model weights')
    args = parser.parse_args()
    print(args)

    hp = op.basename(op.dirname(args.load_model)).split('_')
    args.dataset = hp[0].split('-')[0]
    args.dstype = hp[0].split('-')[1]
    args.model = hp[1]
    args.batchsize = int(hp[3])

    ds2dir = {'pneumonia' : 'PneumoniaCXR', 'CRC' : 'Colorectal', 'covid' : 'Covid', 'breast' : 'BreaKHis400X'}
    if args.dataset in ['pneumonia', 'covid']:
        args.dstype = 'gray'
    elif args.dataset in ['CRC', 'breast']:
        args.dstype = 'color'
    data_path = op.join(args.data_root, ds2dir[args.dataset])
    assert op.exists(data_path)

    out_dir = op.basename(op.dirname(args.load_model))
    # prefix = f'Linear-lr[{args.lrate}]_ep[{args.epochs}]_opt[{args.opti}]'
    out_path = op.join(args.out_root, out_dir)
    
    traindf, valdf, testdf = eval(f'getdf_{ds2dir[args.dataset]}()')
    trainloader, valloader, testloader = get_dataloaders(traindf, valdf, testdf, args.batchsize, args.dstype)
    n_classes = len(np.unique(traindf.iloc[:]['label']))

    model = Model(n_classes)
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model = model.to(device)

    # print(model)

    for mode in ['knn', 'linear']:
        for dl in ['train', 'test']:
            savepath = op.join(out_path, f'tsne_{dl}_{mode}.png')
            data_loader = trainloader if dl == 'train' else testloader
            plot_tsne(model, data_loader, mode, savepath)