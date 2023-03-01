import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score

def train_SSL(epoch, model, data_loader, optimizer, loss_fun, device):
    '''This is for Barlow, SimCLR, VICReg, SimSiam'''
    assert type(model).__name__.split("Model")[0] == type(loss_fun).__name__.split("Loss")[0]
    model.train()
    loss_all = 0.0
    train_bar = tqdm(data_loader)
    for i, batch in enumerate(train_bar):
        x1, x2 = batch['x1'].to(device).float(), batch['x2'].to(device).float()
        z1 = model(x1)
        z2 = model(x2)
        loss = loss_fun(z1, z2)
        loss_all += loss.item()
        train_bar.set_description(f"Epoch: {epoch+1} | Step: [{i+1}/{len(data_loader)}] | Loss: {(loss_all / (i+1)):.6f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_all / len(data_loader)


'''
def train_Tico(epoch, model, C_prev, momentum_val, data_loader, optimizer, loss_fun, device):
    # only for TiCo -- note the extra parameters C_prev, momentum_val
    assert type(model).__name__ == 'TiCoModel'
    assert type(loss_fun).__name__ == 'TiCoLoss'
    model.train()
    loss_all = 0.0
    train_bar = tqdm(data_loader)
    for i, batch in enumerate(train_bar):
        x_query, x_key = batch['x1'].to(device).float(), batch['x2'].to(device).float()
        model.update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        model.update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)
        query = model(x_query)
        key = model.forward_momentum(x_key)
        loss, C = loss_fun(C_prev, query, key)
        C_prev = C.detach()
        loss_all += loss.item()
        train_bar.set_description(f"Epoch: {epoch} | Step: [{i+1}/{len(data_loader)}] | Loss: {(loss_all / (i+1)):.6f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader)
'''

def test_SSL(model, memory_data_loader, test_data_loader, mode, device, k=200):
    assert mode in ['val', 'test']
    model.eval()
    X_train, X_test, y_train, y_test = [], [], [], []
    with torch.no_grad():
        for batch in memory_data_loader:
            data, target = batch['img'], batch['label']
            feature = model.backbone(data.to(device))
            X_train.append(feature)
            y_train.append(target.to(device).long())
        X_train = torch.cat(X_train, dim=0).contiguous() # [N, D]
        y_train = torch.cat(y_train, dim=0).contiguous() # [N,]
        # print(X_train.shape, y_train.shape)
        # extract test set features
        for batch in test_data_loader:
            data, target = batch['img'], batch['label']
            feature = model.backbone(data.to(device))
            X_test.append(feature)
            y_test.append(target.to(device).long())
        X_test = torch.cat(X_test, dim=0).contiguous() # [N, D]
        y_test = torch.cat(y_test, dim=0).contiguous() # [N,]
        # print(X_test.shape, y_test.shape)
    X_train, X_test, y_train, y_test = X_train.cpu().numpy(), X_test.cpu().numpy(), y_train.cpu().numpy(), y_test.cpu().numpy()
    clf = KNN(n_neighbors=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('-'*50 + f"\nKNN (k={k}) Evaluation on {mode}-set:\n" + '-'*50 + "\n")
    print(classification_report(y_test, y_pred, digits=4))
    print('-'*50)
    if mode == 'val':   
        return classification_report(y_test, y_pred, digits=4, output_dict=True), accuracy_score(y_test, y_pred)
    elif mode == 'test':
        return classification_report(y_test, y_pred, digits=4, output_dict=False)
    

def plot_tsne(model, data_loader, device, savepath):
    assert 'png' in savepath 
    model.eval()
    X, y = [], []
    with torch.no_grad():
        for batch in data_loader:
            data, target = batch['img'], batch['label']
            feature = model.backbone(data.to(device))
            X.append(feature)
            y.append(target.to(device).long())
        X = torch.cat(X, dim=0).contiguous() # [N, D]
        y = torch.cat(y, dim=0).contiguous() # [N,]
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500, random_state=16)
    tsne_results = tsne.fit_transform(X)
    colors = plt.cm.get_cmap('inferno', len(np.unique(y)))
    plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y, cmap=colors) 
    plt.savefig(savepath, dpi=300)