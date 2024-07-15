# Main program

# Imports
import sys
import os
os.system('nvidia-smi')


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import argparse
from tqdm import tqdm
from utils.utils import DiffAugment, ParamDiffAug
from utils.network import ConvNet

#!#################################################################################
#!#################################################################################

#* List of Normalizations (Expect testing data to be normalized with):

#* CIFAR100
# mean = [0.5071, 0.4866, 0.4409]
# std = [0.2673, 0.2564, 0.2762]


#* TinyImagenet
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

#!#################################################################################
#!#################################################################################


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



# 
def epoch(mode, dataloader, net, optimizer, criterion, aug, dsa_param, device='cuda'):
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(device)
        if aug:
            img = DiffAugment(img, dsa_strategy, param=dsa_param)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



def evaluator(eval_run, truth_file_path, submission_file_path, gpu_available):

    #! Load the distilled train data:
    #! Assert Data should be unnormalized !

    # if gpu_available:
    device = 'cuda'
    # else:
    #     device = 'cpu'

    data = torch.load(submission_file_path)

    if eval_run == "cifar100":
        num_classes = 100
        assert data.shape[0] == 10*num_classes
        assert data.shape[1] == 3
        assert data.shape[2] == 32
        assert data.shape[3] == 32
        
        

    if eval_run == "tinyimagenet":
        num_classes = 200
        assert data.shape[0] == 10*num_classes
        assert data.shape[1] == 3
        assert data.shape[2] == 64
        assert data.shape[3] == 64
        
    
    images_train = data
    labels_train = torch.tensor([np.ones(10)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1)   
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)
    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=0)

    testdata = torch.load(truth_file_path, map_location='cpu')
    dst_test = TensorDataset(testdata["images_val"], testdata["labels_val"])

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=True, num_workers=0)

    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    param = ParamDiffAug()

    scores = 0
    rounds = 3
    for _ in range(rounds):

        net = ConvNet(channel=3, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=(data.shape[2], data.shape[2]))
        net = net.to(device)
        lr = float(0.01)
        Epoch = int(1000)
        lr_schedule = [Epoch//2+1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss().to(device)

        for ep in tqdm(range(Epoch+1)):
            _, _ = epoch('train', trainloader, net, optimizer, criterion, aug = True, dsa_param=param, device=device)
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        _, acc_test = epoch('test', testloader, net, optimizer, criterion, aug = False, dsa_param=None, device=device)
        
        scores += acc_test
        print(acc_test)
    return scores / rounds



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--truth_dir', type=str, default="./reference_data/")
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument('--submit_dir', type=str, required=True)
    

    args = parser.parse_args()


    # Paths
    truth_dir = args.truth_dir
    output_dir = args.output_dir

    submit_dir = args.submit_dir

    output_file = open(os.path.join(output_dir, 'scores.txt'), 'w')


    score = 0
    gpu_available = torch.cuda.is_available()
    for eval_run in ["cifar100", "tinyimagenet"]:
        submission_file_path = os.path.join(submit_dir, "{}.pt".format(eval_run))
        truth_file_path = "{}{}_test.pt".format(args.truth_dir, eval_run)
        perf = evaluator(eval_run=eval_run, truth_file_path=truth_file_path, submission_file_path=submission_file_path, gpu_available=gpu_available)
        score += perf
        print("Performance on {} dataset is {}".format(eval_run, perf))
    score /= 2
    print("Average Performance is {}".format(score))


    output_file.write("correct: {}".format(score))
    output_file.close()
    print('End of program')



