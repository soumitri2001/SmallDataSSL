import os
import os.path as op
import copy
import argparse
from lossfuns import *
from dataset import *
from util import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImgNetModel(nn.Module):
    def __init__(self, bkb, n_classes):
        super(ImgNetModel, self).__init__()
        self.n_classes = n_classes
        self.bkb = bkb
        self.backbone = torchvision.models.resnet50(pretrained=True) # imagenet weights
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(*[
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, self.n_classes)
        ])
        if self.bkb == 'freeze':
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.classifier(self.backbone(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/soumitri/projects/def-josedolz/soumitri/misc/SmallSSL/data')
    parser.add_argument('--out_root', type=str, default='/home/soumitri/projects/def-josedolz/soumitri/misc/SmallSSL/outputs')
    parser.add_argument('--dataset', type=str, choices=['pneumonia', 'CRC', 'covid', 'breast'], help='choose dataset')
    parser.add_argument('--bkb', type=str, default='train', choices=['freeze', 'train'], help='keep backbone frozen/full model train')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--lrate', type=float, default=1e-3, help='learning rate for model')
    parser.add_argument('--opti', type=str, default='Adam', choices=['SGD', 'Adam', 'LARS'], help='optimizer to be used')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize for imagenet supervised evaluation')
    args = parser.parse_args()
    print(args)

    ds2dir = {'pneumonia' : 'PneumoniaCXR', 'CRC' : 'Colorectal', 'covid' : 'Covid', 'breast' : 'BreaKHis400X'}
    if args.dataset in ['pneumonia', 'covid']:
        args.dstype = 'gray'
    elif args.dataset in ['CRC', 'breast']:
        args.dstype = 'color'
    data_path = op.join(args.data_root, ds2dir[args.dataset])
    assert op.exists(data_path)

    out_dir = '_imagenet_supervised'
    prefix = f'{args.dataset}-ImgNet+bkb[{args.bkb}]-LR[{args.lrate}]_opt[{args.opti}]_BS[{args.batchsize}]'
    out_path = op.join(args.out_root, out_dir)
    
    traindf, valdf, testdf = eval(f'getdf_{ds2dir[args.dataset]}()')
    trainloader, valloader, testloader = get_dataloaders(traindf, valdf, testdf, args.batchsize, args.dstype)
    n_classes = len(np.unique(traindf.iloc[:]['label']))

    model = ImgNetModel(args.bkb, n_classes)
    model = model.to(device)
    lossfun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lrate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    logs = {'epoch' : [], 'trainloss' : [], 'trainacc' : [], 'valloss' : [], 'valacc' : []}

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(args.epochs):
        model.load_state_dict(best_model_wts)
        ### training ###
        model.train()
        loss_all = 0.0
        y_pred, y_test = [], []
        logs['epoch'].append(epoch+1)
        train_bar = tqdm(trainloader)
        for i, batch in enumerate(train_bar):
            data, targets = batch['img'].to(device), batch['label'].to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = lossfun(outputs, targets)
            loss_all += loss.item()
            train_bar.set_description(f"Epoch: {epoch+1} | Step: [{i+1}/{len(trainloader)}] | Loss: {(loss_all / (i+1)):.6f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred.append(preds.to(device).long())
            y_test.append(targets.to(device).long())
        epoch_loss = loss_all / len(trainloader)
        y_pred, y_test = torch.cat(y_pred, dim=0).contiguous().cpu().numpy(), torch.cat(y_test, dim=0).contiguous().cpu().numpy()
        epoch_accuracy = accuracy_score(y_test, y_pred)
        print(f">>> Stats for epoch: {epoch+1} | Train loss: {epoch_loss:.6f} | Train accuracy: {epoch_accuracy:.6f}")
        logs['trainloss'].append(epoch_loss)
        logs['trainacc'].append(epoch_accuracy)

        ### validation ###
        model.eval()
        loss_all = 0.0
        y_pred, y_test = [], []
        with torch.no_grad():
            for i, batch in enumerate(valloader):
                data, targets = batch['img'].to(device), batch['label'].to(device)
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                loss = lossfun(outputs, targets)
                loss_all += loss.item()
                y_pred.append(preds.to(device).long())
                y_test.append(targets.to(device).long())
            epoch_loss = loss_all / len(valloader)
        y_pred, y_test = torch.cat(y_pred, dim=0).contiguous().cpu().numpy(), torch.cat(y_test, dim=0).contiguous().cpu().numpy()
        epoch_accuracy = accuracy_score(y_test, y_pred)
        print(f">>> Stats for epoch: {epoch+1} | Val loss: {epoch_loss:.6f} | Val accuracy: {epoch_accuracy:.6f}")
        logs['valloss'].append(epoch_loss)
        logs['valacc'].append(epoch_accuracy)
        if epoch_accuracy >= best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            ckpt = {
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'epochs' : epoch
            }
            torch.save(ckpt, op.join(out_path, f'{prefix}_checkpoint.pt'))
            torch.save(model.state_dict(), op.join(out_path, f'{prefix}_clsmodel.pth'))

        if epoch > 25:
            scheduler.step()
    
        pd.DataFrame(logs).to_csv(op.join(out_path, f"{prefix}_trainvallogs.csv"), index=False)

    ### testing ###
    model.load_state_dict(best_model_wts)
    model.eval()
    y_pred, y_test = [], []
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            data, targets = batch['img'].to(device), batch['label'].to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = lossfun(outputs, targets)
            loss_all += loss.item()
            y_pred.append(preds.to(device).long())
            y_test.append(targets.to(device).long())
        epoch_loss = loss_all / len(testloader)
    y_pred, y_test = torch.cat(y_pred, dim=0).contiguous().cpu().numpy(), torch.cat(y_test, dim=0).contiguous().cpu().numpy()
    cls_report = classification_report(y_test, y_pred, digits=4, output_dict=False)
    outfile = open(op.join(out_path, f'{prefix}-TestSet.txt'), 'w')
    outfile.write(f"Testset report | ImageNet eval | {args.epochs} epochs \n\n")
    outfile.write(cls_report)
    outfile.close()

    # plot_tsne(model, trainloader, device, f'{out_path}/tsne-train.png')
    # plot_tsne(model, testloader, device, f'{out_path}/tsne-test.png')

    print('Model and logs saved -- Imagenet supervised evaluation complete!!')
    

if __name__ == '__main__':
    main()