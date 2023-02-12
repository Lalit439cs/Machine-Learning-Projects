import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir_path")
args = parser.parse_args()
dataset_dir_path = args.dataset_dir_path

#libraries
from time import time
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

# import torch
from torch.utils.data import Dataset
# from torchvision import datasets
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image

img_dir = dataset_dir_path+'/images/images/'
train_path_x = dataset_dir_path+'/train_x.csv'
train_path_y = dataset_dir_path+'/train_y.csv'
test_path_x = dataset_dir_path+'/non_comp_test_x.csv'
test_path_y = dataset_dir_path+'/non_comp_test_y.csv'

class CustomImageDataset(Dataset):
	def __init__(self, train_path_x, train_path_y, img_dir, transform=None, target_transform=None):
		self.img_labels = pd.read_csv(train_path_y)
		self.img_names  = pd.read_csv(train_path_x)
		self.img_names  = self.img_names.set_index('Id')
		self.img_labels = self.img_labels.set_index('Id')
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

		def __len__(self):
			return len(self.img_labels)

		def __getitem__(self, idx):        
			img_path = os.path.join(self.img_dir, self.img_names['Cover_image_name'][idx])
			image = read_image(img_path)
			image = image/255.0

			label = self.img_labels['Genre'][idx]
			if self.transform:
			image = self.transform(image)
			if self.target_transform:
			label = self.target_transform(label)

			#         device = "cuda" if torch.cuda.is_available() else "cpu"
			#         image.to(device), label
			return image, label

training_data = CustomImageDataset(train_path_x, train_path_y, img_dir)
test_data = CustomImageDataset(test_path_x, test_path_y, img_dir)
train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)
test_dataloader  = DataLoader(test_data, batch_size=100, shuffle=True)
#try dataloader on cuda


class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()        
			self.network = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=(5,5)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,2), stride=2),
			nn.Conv2d(32, 64, kernel_size=(5,5)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,2), stride=2),
			nn.Conv2d(64, 128, kernel_size=(5,5)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,2), stride=2),
			nn.Flatten(),
			nn.Linear(128*24*24, 128),
			nn.ReLU(),
			nn.Linear(128, 30)
		)

	def forward(self, x):
		return self.network(x) 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
# print(model)
# print(device)

learning_rate = 0.1 #1e-2
loss_fn = nn.CrossEntropyLoss()#torch.nn
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def accuracy(dataloader, model):    
	cnt = 0
	device = "cuda" if torch.cuda.is_available() else "cpu"
	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X.to(device))
			cnt += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()#to(device)
	return (cnt/len(dataloader.dataset))*100

def train(dataloader, test_dataloader,model, loss_fn, optimizer,  epochs=12, batch_size=100, set_epsilon=False, epsilon=1e-5):
	loss_vals, epoch_vals, accuracy_vals,test_accuracies = [], [], [],[]
	device = "cuda" if torch.cuda.is_available() else "cpu"
	for epoch in range(epochs):
		for batch, (X, y) in enumerate(dataloader):            
			pred = model(X.to(device))
			#             print(type(pred.to(device)))
			#             print(device)
			loss = loss_fn(pred, y.to(device))
			optimizer.zero_grad()#above
			loss.backward()
			optimizer.step()

		loss_vals.append(loss.item())
		epoch_vals.append(epoch)
		accuracy_vals.append(accuracy(dataloader, model))
		test_accuracies.append(accuracy(test_dataloader, model))
		print('epoch = {}'.format(epoch))
		print('train accuracy = {}'.format(accuracy_vals[-1]))
		print('test accuracy = {}'.format(test_accuracies[-1]))
	return loss_vals, epoch_vals, accuracy_vals,test_accuracies

# t1=time()
loss_vals, epoch_vals,train_accuracies,test_accuracies = train(train_dataloader,test_dataloader, model, loss_fn, optimizer, epochs=18)#epochs=15 fine
# t=time()-t1
# print('time',t)
#time calc
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# fig, ax = plt.subplots(figsize = (12,12))
# ax.set_xlabel("epoch-")
# ax.set_ylabel("Accuracy")
# ax.set_title("epochs vs Accuracy")
# ax.plot(epoch_vals, train_accuracies, marker="o", label="train", drawstyle="steps-post")
# ax.plot(epoch_vals, test_accuracies, marker="D", label="test", drawstyle="steps-post")
# ax.legend()
# pt.show()
# fig.savefig('epoch_vs_accuracy.png')
# plt.plot(epoch_vals, accuracy_vals)
# plt.savefig('epoch_vs_accuracy.png') 

def save_preds(dataLoader, model):
	cnt = 0
	device = "cuda" if torch.cuda.is_available() else "cpu"
	all_preds = []
	with torch.no_grad():
		for X, y in dataLoader:
			outputs = model(X.to(device))
			preds = outputs.argmax(1)
			preds = preds.detach().cpu().numpy()
			preds = list(preds)
			for pred in preds:
				all_preds.append(preds)
	df = pd.DataFrame(data=all_preds)
	df.to_csv('non_comp_test_pred_y.csv', index=False)
	return

save_preds(test_dataloader, model)