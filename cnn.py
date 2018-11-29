from __future__ import print_function

import numpy
import time
from PIL import Image


import matplotlib as mpl; mpl.use('Agg')    
#It is important to set the mode before import pyplot, otherwise we cannot graph
import matplotlib.pyplot as plt

import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

#hack
torchvision.datasets.folder.IMG_EXTENSIONS.append('.tif')



class Net(nn.Module):

    fc_layers = 9600 #typically 9600 
    channels=3
    output=7

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(self.channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=5)
        self.fc1 = nn.Linear(self.fc_layers, 50) #seven output classes
        self.fc2 = nn.Linear(50, self.output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(-1, self.fc_layers)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def my_loader(path):
    return Image.open(path)

def mnist_loader(workers=1, train_bool=True):
    return torch.utils.data.DataLoader(
            datasets.MNIST("/nfs/stak/users/kramerer/diatom/Diatom_data/raw_data/data",
                train=train_bool,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,))
                    ])),
                batch_size=1, shuffle=True,
            )



#image = image / image.max()
def gen_data_loader(workers=1, train_bool=True, ):


    #expects a directory where each sub directory is a class
    #The path variable is Top-level to train/test which contain this format

    path = '/nfs/stak/users/kramerer/diatom/Diatom_data/raw_data/grouped'
    if train_bool:
        path += '/train'
    else:
        path += '/test'

    transform_set = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x/ torch.max(x)),
                #transforms.Normalize(),
            ])

    image_data = torchvision.datasets.ImageFolder(
            path,
            transform=transform_set
            )


    data_loader = torch.utils.data.DataLoader(
            image_data,
            batch_size=1,
            shuffle=True,
            num_workers=workers
    )
    #import pdb; pdb.set_trace();
    return data_loader



def target_cast(target):
    return target
    #return torch.Tensor([target]).long()


def dump_graph():
    #accept a unique name or generate one based on date/time
    pass

def train(model, data_loader, opt, epochs):

    test_epochs = []
    train_epochs = []

    test_loader = gen_data_loader(train_bool=False)
    model.train()
    for e in range(epochs):
        print("Starting Epoch: ", e+1)

        for (data, target) in data_loader:
            if mode=="gpu":
                data, target = Variable(data).cuda(), Variable(target_cast(target)).cuda()
            else:
                data, target = Variable(data), Variable(target_cast(target))

            opt.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            opt.step()
           
        print("="*30)
        print("Test Accuracy: ")
        test_epochs.append(test(model, test_loader))
        print("="*30)
        print("Train Accuracy: ")
        train_epochs.append(test(model, data_loader))
        print("="*30)
    print(test_epochs)
    print(train_epochs)
    import pdb; pdb.set_trace()


def test(model, data_loader):
    model.eval()  # Set network to evaluation mode
    #test_loss = 0

    correct = 0

    guess_bin = [0]*Net.output
    correct_bin = [0]*Net.output

    total = len(data_loader)

    for data, target in data_loader:
        if mode=="gpu":
            data, target = Variable(data).cuda(), Variable(target_cast(target)).cuda()
        else:
            data, target = Variable(data), Variable(target_cast(target))

        output = model(data)
        _, pred = torch.max(output.data, 1)
        #test_loss += F.cross_entropy(output, 
        #        target).item()
        correct += (pred == target.data).sum()
        guess_bin[int(pred)] += 1
        for guess,real in zip(pred, target.data):
            if guess==real:
                correct_bin[guess] +=1

       # print("prediction: ", pred)

    print("Total correct: ", correct)
    print("Total examples: ", total)
    print("Accuracy: ", float(correct)/float(total))
    print("Guess bin: ", guess_bin)
    print("Correct bin: ", correct_bin)

    return float(correct)/float(total)


#assumes 0-epoch
def graph_plotter(data_sets, y_labels, epochs, title="default_title"):
    #Plot each dataset, assume they have similar X components for now\ 
    #epochs is unused.

    for s, l in zip(data_sets, y_labels):
        plt.show(s, label=l)

    #Move legend
    plt.legend(loc="upper left")

    #add a unique value so that we don't overwrite an old image
    plt.savefig("img/" + title + time.time() + ".png")


def main():

    print("Loading Data")
    loader = gen_data_loader()

    if mode == "gpu":
        model = Net().cuda()  # where mode is "cuda" or "cpu"
    else:
        model = Net()
 
    print("Model Initialized")

    optimizer = optim.SGD(model.parameters(), lr=.001, weight_decay=.0001, momentum=.5)

    #optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)


    print("Training Model")
    train(model, loader, optimizer, 50)
    print("Training complete")

    test_loader = gen_data_loader(train_bool=False)
    test(model, test_loader)
    
    print("Now testing train accuracy")
    test(model, loader)

    print("Testing complete!")



if __name__ == '__main__':
    import sys;
    mode = "gpu"
    if len(sys.argv) > 2:
        #anything other than the string "gpu" is treated as cpu
        mode = sys.argv[1] or mode
    if len(sys.argv) > 3:
        #again more command line hacks, this is just checking to see if we get
        #a third argument, in which case we assume that you meant to use the
        #mnist dataset. This is used for Erich's testing not much else. 
        print("Using MNIST dataset.")
        gen_data_loader = mnist_loader
        Net.channels=1
        Net.fc_layers = 160
        Net.output = 10
    main()


