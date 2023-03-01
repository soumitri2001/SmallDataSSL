import os
import json
import argparse
from lossfuns import *
from dataset import *
from util import *
from model import *
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/soumitri/projects/def-josedolz/soumitri/misc/SmallSSL/data')
    parser.add_argument('--out_root', type=str, default='/home/soumitri/projects/def-josedolz/soumitri/misc/SmallSSL/outputs')
    parser.add_argument('--dataset', type=str, choices=['pneumonia', 'CRC', 'covid', 'breast'], help='choose dataset')
    parser.add_argument('--dstype', type=str, choices=['color', 'gray'], help='dataset is color/grayscale')
    parser.add_argument('--model', type=str, choices=['SimCLR', 'VICReg', 'Barlow', 'SimSiam', 'DCLW'], help='choose SSL model')
    parser.add_argument('--epochs', type=int, default=250, help='training epochs')
    parser.add_argument('--batchsize', type=int, default=64, help='batchsize to be used')
    parser.add_argument('--lrate', type=float, default=1e-2, help='learning rate to be used')
    parser.add_argument('--opti', type=str, default='SGD', choices=['SGD', 'Adam', 'LARS'], help='optimizer to be used')
    parser.add_argument('--knn', type=int, default=200, help='K-value for KNN')
    args = parser.parse_args()
    print(args)

    ds2dir = {'pneumonia' : 'PneumoniaCXR', 'CRC' : 'Colorectal', 'covid' : 'Covid', 'breast' : 'BreaKHis400X'}
    
    if args.dstype is None:
        if args.dataset in ['pneumonia', 'covid']:
            args.dstype = 'gray'
        elif args.dataset in ['CRC', 'breast']:
            args.dstype = 'color'
            
    data_path = os.path.join(args.data_root, ds2dir[args.dataset])
    assert os.path.exists(data_path)

    out_dir = f'{args.dataset}-{args.dstype}_{args.model}_{args.epochs}_{args.batchsize}_{args.lrate}_{args.opti}'
    out_path = os.path.join(args.out_root, out_dir)
    os.makedirs(out_path, exist_ok=True)

    traindf, valdf, testdf = eval(f'getdf_{ds2dir[args.dataset]}()')
    trainloader, valloader, testloader = get_dataloaders(traindf, valdf, testdf, args.batchsize, args.dstype)
    
    model = eval(f'{args.model}Model().to(device)')
    optimizer = eval(f'optim.{args.opti}(params=model.parameters(), lr=args.lrate)')
    lossfun = eval(f'{args.model}Loss()')

    logs = []
    best_acc = 0.0
    print(f'\nStarting training model: {args.model} | {args.opti} | {args.epochs} epochs\n')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = train_SSL(epoch, model, trainloader, optimizer, lossfun, device)
        model.eval()
        cls_report, acc_score = test_SSL(model, trainloader, valloader, 'val', device, args.knn)
        acc, wt_pre, wt_rec, wt_f1 = cls_report['accuracy'], cls_report['weighted avg']['precision'], cls_report['weighted avg']['recall'], cls_report['weighted avg']['f1-score']
        logs.append({'epoch' : (epoch+1), 'loss' : epoch_loss, 'accuracy' : acc, 'precision' : wt_pre, 'recall' : wt_rec, 'f1_score' : wt_f1})
        ckpt = {'model' : model.state_dict(), 'optimizer' : optimizer.state_dict(), 'epochs' : epoch}
        if acc_score >= best_acc:
            best_acc = acc_score
            torch.save(model.backbone.state_dict(), os.path.join(out_path, 'encoder.pth'))
        torch.save(ckpt, os.path.join(out_path, 'checkpoint.pt'))
        pd.DataFrame(logs).to_csv(os.path.join(out_path, "trainvallogs_KNN.csv"), index=False)

    ### testing ###        
    model.eval()
    cls_report = test_SSL(model, trainloader, testloader, 'test', device, args.knn)
    outfile = open(os.path.join(out_path, 'test_KNN.txt'), 'w')
    outfile.write(f"Testset report | {args.model} pretraining | {args.epochs} epochs | KNN (K={args.knn})\n\n")
    outfile.write(cls_report)
    outfile.close()

    # plot_tsne(model, trainloader, device, f'{out_path}/tsne-train.png')
    # plot_tsne(model, testloader, device, f'{out_path}/tsne-test.png')

    print('Model and logs saved -- training complete!!')









    
    
    