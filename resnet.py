WANDB=False
import os
import gc
import torch
import shutil
import tempfile
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import torchvision as tv
import torch.optim as optim
from collections import OrderedDict
from torch.autograd import Variable
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score
from torchvision import datasets, transforms, models
import wandb
import warnings
warnings.filterwarnings("ignore")

if WANDB:
    wandb.init(
            project="MultimodalCommentAnalysis",
            name="resnet18",
            )
    

data_dir = './data/'
class_names0 = os.listdir(data_dir)

class_names=[]
for item in class_names0:
    class_names+=[item]
num_class = len(class_names)
image_files = [[os.path.join(data_dir, class_name, x) \
                for x in os.listdir(os.path.join(data_dir, class_name))[:16000]] \
                for class_name in class_names]

image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
num_total = len(image_label_list)

width, height = Image.open(image_file_list[0]).size

print("Total image count:", num_total)
print("Image dimensions:", width, "x", height)
print("Label names:", class_names)
print("Label counts:", [len(image_files[i]) for i in range(num_class)])

valid_frac = 0.3
trainX, trainY = [], []
valX, valY = [], []
testX, testY = [], []

if not os.path.exists('./data_train'):
    print("Create dataset...")
    for i in tqdm(range(num_total)):
        rann = np.random.random()
        if rann < valid_frac:
            valX.append(image_file_list[i])
            valY.append(image_label_list[i])
            j = image_file_list[i]
            k = j.split('/')[-2]
            r = j.split('/')[-1]
            os.makedirs(f'./data_val/{k}', exist_ok=True)
            shutil.copyfile(j, os.path.join(f'./data_val/{k}/', r))
        else:
            trainX.append(image_file_list[i])
            trainY.append(image_label_list[i])
            j = image_file_list[i]
            k = j.split('/')[-2]
            r = j.split('/')[-1]
            os.makedirs(f'./data_train/{k}', exist_ok=True)
            shutil.copyfile(j, os.path.join(f'./data_train/{k}/', r))
else:
    trainX = os.listdir('./data_train')
    valX = os.listdir('./data_val')
          
print(len(trainX), len(valX))

train_dir = './data_train'
val_dir = './data_val'
batch_size = 64

transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomAffine(degrees=(10, 150), translate=(0.2, 0.5), shear=45),
     transforms.RandomHorizontalFlip(),
     transforms.Resize([224, 224]),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
     ])

transform_val = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize([224, 224]),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
     ])

image_datasets = {}
image_datasets["train"] = datasets.ImageFolder(root = train_dir, transform=transform_train)
image_datasets["valid"] = datasets.ImageFolder(root = val_dir, transform=transform_val)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

print(class_names)

train_loader = torch.utils.data.DataLoader(image_datasets["train"], sampler=ImbalancedDatasetSampler(image_datasets["train"]), batch_size=batch_size,
                                          num_workers = 12)
valid_loader = torch.utils.data.DataLoader(image_datasets["valid"], sampler=ImbalancedDatasetSampler(image_datasets["valid"]), batch_size=batch_size,
                                          num_workers = 12)

print(dataset_sizes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


parser = argparse.ArgumentParser()

parser.add_argument('-save_dir', action="store", dest="save_dir", type=str, default="./checkpoint/")
parser.add_argument('-lr', action="store", dest="lr", type=float, default=1e-4)
parser.add_argument('-hiddenunits', action="store", dest="hiddenunits", type=int, default=128)
parser.add_argument('-epochs', action="store", dest="epochs", type=int, default=10)
ins=parser.parse_args(args=[])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
        self.backbone = tv.models.resnet18()
        self.fc1 = nn.Linear(1000, ins.hiddenunits)
        self.fc2 = nn.Linear(ins.hiddenunits, 2)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

        
model = Model()
print("Model:\n", model)

gc.collect()
torch.cuda.empty_cache()

use_gpu = torch.cuda.is_available()
print("Use GPU: ", use_gpu)

steps = int(len(trainX) / batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = ins.lr)
scaler = GradScaler()

model.cuda()

train_loss=[]
test_loss=[]
best_acc = 0

print("Training...")
# Training
for epoch in range(ins.epochs):
    # Reset variables at 0 epoch
    correct=0
    iteration=0
    iter_loss=0.0
    accuracy_list = []
    train_loss_list = []
    train_acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []
    mse_list = []
    mae_list = []
  
    model.train() # training mode

    with tqdm(total=len(train_loader)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch, ins.epochs - 1))
        for i,(inputs,label) in enumerate(train_loader):
            labels = F.one_hot(label, num_classes=2).type(torch.FloatTensor)
            inputs=Variable(inputs)
            labels=Variable(labels)
            cuda=torch.cuda.is_available()
            if cuda:
                inputs=inputs.cuda()
                labels=labels.cuda()
            optimizer.zero_grad() # clear gradient
            with autocast():
                pred=model(inputs)
                loss=criterion(pred,labels)
                loss = loss.mean()
            iter_loss += loss.item() # accumulate loss
            loss.requires_grad_(True)
            scaler.scale(loss).backward() # backpropagation
            scaler.step(optimizer)
            scaler.update()

            
            if WANDB:
                wandb.log({"train_loss": loss.item(),})

            # save the correct predictions for training data
            accuracy = accuracy_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
            precision = precision_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            recall = recall_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            f1 = f1_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            try:
                roc_auc = roc_auc_score(labels.argmax(dim=1).cpu().numpy(), F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1])
            except ValueError:
                pass
            mse = mean_squared_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().detach().numpy())
            mae = mean_absolute_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().detach().numpy())
            
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            roc_auc_list.append(roc_auc)
            mse_list.append(mse)
            mae_list.append(mae)
            
            correct += accuracy
            iteration +=1
            
            _tqdm.set_postfix(loss='{:.3f}'.format(loss), accuracy='{:.3f}'.format(accuracy),
                    lr='{:.1e}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
            _tqdm.update(1)

    train_loss.append(iter_loss/iteration)
    
    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_roc_auc = np.mean(roc_auc_list)
    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)
    if WANDB:
        wandb.log({
            "train_accuracy":avg_accuracy,
            "train_precision":avg_precision,
            "train_recall":avg_recall,
            "train_f1":avg_f1,
            "train_roc_auc":avg_roc_auc,
            "train_mse":avg_mse,
            "train_mae":avg_mae,
            })
    print({
        "train_accuracy":avg_accuracy,
        "train_precision":avg_precision,
        "train_recall":avg_recall,
        "train_f1":avg_f1,
        "train_roc_auc":avg_roc_auc,
        "train_mse":avg_mse,
        "train_mae":avg_mae,
        })
  
    correct=0
    iteration=0
    valid_loss=0.0
    print("testing...")
  
    model.eval()
    accuracy_list = []
    train_loss_list = []
    train_acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []
    mse_list = []
    mae_list = []
    test_accuracy = []
  
    with tqdm(total=len(valid_loader)) as _tqdm:
        for i, (inputs, label) in enumerate(valid_loader):
            labels = F.one_hot(label, num_classes=2).type(torch.FloatTensor)
            inputs=Variable(inputs)
            labels=Variable(labels)

            cuda=torch.cuda.is_available()
            if cuda:
                inputs=inputs.cuda()
                labels=labels.cuda()
            with torch.no_grad():
                pred=model(inputs)
                loss=criterion(pred,labels)
                loss=loss.mean()
                valid_loss += loss.item()
                iteration+=1  
                
                accuracy = accuracy_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
                precision = precision_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
                recall = recall_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
                f1 = f1_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
                try:
                    roc_auc = roc_auc_score(labels.argmax(dim=1).cpu().numpy(), F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1])
                except ValueError:
                    pass
                mse = mean_squared_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
                mae = mean_absolute_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())

                accuracy_list.append(accuracy)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                roc_auc_list.append(roc_auc)
                mse_list.append(mse)
                mae_list.append(mae)
            _tqdm.update(1)

    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_roc_auc = np.mean(roc_auc_list)
    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)
    if WANDB:
        wandb.log({
            "val_accuracy":avg_accuracy,
            "val_precision":avg_precision,
            "val_recall":avg_recall,
            "val_f1":avg_f1,
            "val_roc_auc":avg_roc_auc,
            "val_mse":avg_mse,
            "val_mae":avg_mae,
            })
    print({
        "val_accuracy":avg_accuracy,
        "val_precision":avg_precision,
        "val_recall":avg_recall,
        "val_f1":avg_f1,
        "val_roc_auc":avg_roc_auc,
        "val_mse":avg_mse,
        "val_mae":avg_mae,
        })
    
    test_loss.append(valid_loss/iteration)
    test_accuracy.append((100*correct/len(image_datasets["valid"])))
    
    state_dict = model.module.state_dict() if next(model.parameters()).device == 'cuda:0' else model.state_dict()
    torch.save({'epoch': epoch, 'model_state_dict': state_dict},
            f'./{ins.save_dir}/model_epoch_{epoch+1}.pth')
