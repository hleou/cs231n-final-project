import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim

import time
import os
import copy
import pickle
from PIL import Image
from torchvision import models, transforms

with open("data/anno/img_category.txt", 'r') as f:
    label_lines = f.readlines()[2:]
f.close()

with open("data/eval/eval_partition.txt", 'r') as f:
    set_type_lines = f.readlines()[2:]
f.close()

num_images = len(label_lines)
image_files = {'train':[], 'test':[], 'val':[]}
image_labels = {'train':[], 'test':[], 'val':[]}
    
for i in range(num_images):
    set_type = set_type_lines[i].split()[1]
    label = label_lines[i].split()[1]
    image = label_lines[i].split()[0]
    image_files[set_type].append(image)
    image_labels[set_type].append(label)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_files, img_labels):
        self.root_dir = root_dir
        self.img_files = img_files
        self.img_labels = img_labels
        
    def __len__(self):
        return len(self.img_files)
    
    def toTensor(self, image):
        toTensor_ = transforms.ToTensor()
        return toTensor_(image)
    
    def pad(self, image):
        size = np.asarray(image.size)
        pad_dim = (np.max(size) - size)//2
        pad_ = transforms.Pad(tuple(pad_dim), fill=(255,255,255))
        return pad_(image)
    
    def resize(self, image):
        resize_ = transforms.Resize((224,224))
        return resize_(image)
    
    def norm(self, image):
        mean, std = image.mean(dim=(1,2)), image.std(dim=(1,2))
        normalize_ = transforms.Normalize(mean, std)
        return normalize_(image)
    
    def __getitem__(self, idx):
        image_name = self.root_dir + self.img_files[idx]
        image = self.pad(Image.open(image_name))
        image = self.resize(image)
        image = self.toTensor(image)
        image = self.norm(image)
        label = self.img_labels[idx]
        return (image, int(label)-1)

datasets = {}
dataloaders = {}

for key in image_files:
    datasets[key] = Dataset('data/', image_files[key], image_labels[key])
    print(key +  ' size:', len(datasets[key]))
    dataloaders[key] = torch.utils.data.DataLoader(datasets[key], batch_size=32, shuffle=True, num_workers=8)
    
def ResNetModel():
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    num_classes = 50
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Softmax())
    
    return model_ft.to(device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnetModel = ResNetModel()

def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            i = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #keep from falling asleep
                if(i%100==0): print('.', end='', flush=True)
                i += 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, 'models/resnet50-100epoch.pt')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer_ft = optim.Adam(resnetModel.parameters(), lr=1e-4, weight_decay=1e-4)

# Learning Rate Scheduler
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

model = train_model(resnetModel, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)
torch.save(model, 'models/resnet50-100epoch-final.pt')
