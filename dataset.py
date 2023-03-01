import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def getdf_BreaKHis400X():
    root = "./data/BreaKHis400X"
    print('-'*50 + f'\nExtracting data from {root}\n' + '-'*50)
    traindf, testdf = pd.DataFrame(columns=['filepath', 'label']), pd.DataFrame(columns=['filepath', 'label'])
    label_encode = {'benign' : 0, 'malignant' : 1}
    for label in label_encode.keys():
        for f in os.listdir(root+"/train/"+label):
            fpath = os.path.join(root, 'train', label, f)
            traindf = traindf.append({'filepath' : fpath, 'label' : label_encode[label]}, ignore_index=True)
        for f in os.listdir(root+"/test/"+label):
            fpath = os.path.join(root, 'test', label, f)
            testdf = testdf.append({'filepath' : fpath, 'label' : label_encode[label]}, ignore_index=True)
    traindf, testdf = traindf.sample(frac=1), testdf.sample(frac=1)
    traindf, valdf = train_test_split(traindf, test_size=0.1, shuffle=True, random_state=16)
    print(f"Trainset: {len(traindf)} images | Valset: {len(valdf)} images | Testset: {len(testdf)} images")
    return traindf, valdf, testdf

def getdf_Covid():
    root = "./data/Covid"
    print('-'*50 + f'\nExtracting data from {root}\n' + '-'*50)
    traindf, testdf = pd.DataFrame(columns=['filepath', 'label']), pd.DataFrame(columns=['filepath', 'label'])
    label_encode = {'Covid' : 0, 'Normal' : 1, 'Viral Pneumonia' : 2}
    for label in label_encode.keys():
        for f in os.listdir(root+"/train/"+label):
            fpath = os.path.join(root, 'train', label, f)
            traindf = traindf.append({'filepath' : fpath, 'label' : label_encode[label]}, ignore_index=True)
        for f in os.listdir(root+"/test/"+label):
            fpath = os.path.join(root, 'test', label, f)
            testdf = testdf.append({'filepath' : fpath, 'label' : label_encode[label]}, ignore_index=True)
    traindf, testdf = traindf.sample(frac=1), testdf.sample(frac=1)
    traindf, valdf = train_test_split(traindf, test_size=0.1, shuffle=True, random_state=16)
    print(f"Trainset: {len(traindf)} images | Valset: {len(valdf)} images | Testset: {len(testdf)} images")
    return traindf, valdf, testdf

def getdf_Colorectal():
    root = "./data/Colorectal"
    print('-'*50 + f'\nExtracting data from {root}\n' + '-'*50)
    df = pd.DataFrame(columns=['filepath', 'label'])
    label_encode = {f:i for i,f in enumerate(sorted(os.listdir(root)))}
    for label in label_encode.keys():
        for f in os.listdir(root+"/"+label):
            fpath = os.path.join(root, label, f)
            df = df.append({'filepath' : fpath, 'label' : label_encode[label]}, ignore_index=True)   
    df = df.sample(frac=1)
    traindf, testdf = train_test_split(df, test_size=0.2, shuffle=True, random_state=16)
    traindf, valdf = train_test_split(traindf, test_size=0.1, shuffle=True, random_state=16)
    print(f"Trainset: {len(traindf)} images | Valset: {len(valdf)} images | Testset: {len(testdf)} images")
    return traindf, valdf, testdf

def getdf_PneumoniaCXR():
    root = "./data/PneumoniaCXR"
    print('-'*50 + f'\nExtracting data from {root}\n' + '-'*50)
    traindf, testdf = pd.DataFrame(columns=['filepath', 'label']), pd.DataFrame(columns=['filepath', 'label'])
    label_encode = {'NORMAL' : 0, 'PNEUMONIA' : 1}
    for label in label_encode.keys():
        for dir in ['train', 'val']:
            for f in os.listdir(root+"/"+dir+"/"+label):
                fpath = os.path.join(root, dir, label, f)
                traindf = traindf.append({'filepath' : fpath, 'label' : label_encode[label]}, ignore_index=True)
        for f in os.listdir(root+"/test/"+label):
            fpath = os.path.join(root, 'test', label, f)
            testdf = testdf.append({'filepath' : fpath, 'label' : label_encode[label]}, ignore_index=True)
    traindf, testdf = traindf.sample(frac=1), testdf.sample(frac=1)
    traindf, valdf = train_test_split(traindf, test_size=0.1, shuffle=True, random_state=16)
    print(f"Trainset: {len(traindf)} images | Valset: {len(valdf)} images | Testset: {len(testdf)} images")
    return traindf, valdf, testdf


class MedDataset(Dataset):
    def __init__(self, df, mode, dstype):
        super(MedDataset, self).__init__()
        self.df = df
        self.mode = mode
        self.dstype = dstype
        mean = [0.485, 0.456, 0.406] if self.dstype == 'color' else [0.5, 0.5, 0.5] # gray
        std = [0.229, 0.224, 0.225] if self.dstype == 'color' else [0.5, 0.5, 0.5] # gray
        
        self.basic_transforms = transforms.Compose([transforms.Resize((224,224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=mean, std=std)
                                                                        ])
        self.augment_transforms = transforms.Compose([   
                transforms.RandomResizedCrop(224, scale=(0.08, 0.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
                transforms.RandomSolarize(threshold=128, p=0.1), #128 taken from BYOL
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.df)

    def __augment__(self, x):
        return self.augment_transforms(x)

    def __getitem__(self, idx):
        sample = {}
        imgpath, label = self.df.iloc[idx]['filepath'], self.df.iloc[idx]['label']
        orgimg = Image.open(imgpath).convert('RGB')
        augimg1, augimg2 = self.__augment__(orgimg), self.__augment__(orgimg)
        orgimg = self.basic_transforms(orgimg)
        if self.mode == 'train':
            sample = {'img' : orgimg, 'x1' : augimg1, 'x2' : augimg2, 'label' : label}
        elif self.mode == 'val' or self.mode == 'test':
            sample = {'img' : orgimg, 'label' : label}
        return sample

    
def get_dataloaders(train_df, val_df, test_df, batchsize, dstype):
    trainset, valset, testset = MedDataset(train_df, 'train', dstype), MedDataset(val_df, 'val', dstype), MedDataset(test_df, 'test', dstype)
    trainloader, valloader, testloader = DataLoader(trainset, batch_size=batchsize, shuffle=True), DataLoader(valset, batch_size=1, shuffle=False), DataLoader(testset, batch_size=1, shuffle=False)
    return trainloader, valloader, testloader