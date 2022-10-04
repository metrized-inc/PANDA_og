import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm

from data import create_datamodule
from utils_custom.settings_functions import load_settings
from mvtec_ad import MVTecAD
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def _convert_label(x):
    '''
    convert anomaly label. 0: normal; 1: anomaly.
    :param x (int): class label
    :return: 0 or 1
    '''
    return 0 if x == 0 else 1

def train_model(model, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, feature_space = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):

        # imgs, _ = batch
        imgs, _, _, _ = batch
        # imgs, mask, target = batch

        images = imgs.to(device)

        optimizer.zero_grad()

        _, features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='Train set feature extracting'):
            # imgs, _ = batch
            imgs, _, _, _ = batch
            # imgs, mask, target = batch

            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        test_labels = []
        for batch in tqdm(test_loader, desc='Test set feature extracting'):
            # imgs, _ = batch
            imgs, y, _, _ = batch
            # imgs, mask, target = batch

            imgs = imgs.to(device)
            _, features = model(imgs)
            test_feature_space.append(features)
            test_labels.extend(y.type(torch.int).tolist())
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        # test_labels = test_loader.dataset.targets

        test_labels = [0 if x > 0 else 1 for x in test_labels]

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)

    ewc_loss = None

    # Freezing Pre-trained model for EWC
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        ewc_loss = EWCLoss(frozen_model, fisher)

    utils.freeze_parameters(model)
    # train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)

    settings_path = "panda.hjson"
    settings = load_settings(settings_path)
    dm = create_datamodule(settings)
    train_loader = dm.train_dataloader()
    test_loader = dm.val_dataloader()

    # define transforms
    # transform = transforms.Compose([transforms.Resize(256),
    #                                   transforms.CenterCrop(224),
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    # transform_gray = transforms.Compose([
    #                              transforms.Resize(256),
    #                              transforms.CenterCrop(224),
    #                              transforms.Grayscale(num_output_channels=3),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                             ])

    # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # target_transform = transforms.Lambda(_convert_label)

    # mvtec_train = MVTecAD('data',
    #                 subset_name='bottle',
    #                 train=True,
    #                 transform=transform,
    #                 mask_transform=transform,
    #                 target_transform=target_transform,
    #                 download=True)

    # mvtec_test = MVTecAD('data',
    #                 subset_name='bottle',
    #                 train=False,
    #                 transform=transform,
    #                 mask_transform=transform,
    #                 target_transform=target_transform,
    #                 download=True)

    # # feed to data loader
    # train_loader = DataLoader(mvtec_train,
    #                          batch_size=32,
    #                          shuffle=True,
    #                          num_workers=0,
    #                          pin_memory=True,
    #                          drop_last=True)

    # test_loader = DataLoader(mvtec_test,
    #                          batch_size=32,
    #                          shuffle=True,
    #                          num_workers=0,
    #                          pin_memory=True,
    #                          drop_last=True)

    train_model(model, train_loader, test_loader, device, args, ewc_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    main(args)
