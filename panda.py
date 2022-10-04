from re import S
import numpy as np
import torch
import os
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss, ClassAccuracy
import utils.panda_utils as utils
from copy import deepcopy
from tqdm import tqdm

from utils.settings_functions import *
from data import create_datamodule

def train_model(model, train_loader, val_loader, device, args, ewc_loss, save_path, project):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, val_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))

    # try:
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, feature_space = get_score(model, device, train_loader, val_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))

        ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                'auc': auc}

        if (epoch+1) == args.epochs:
            torch.save(ckpt, os.path.join(save_path, 'last.ckpt'))
        elif (epoch+1) % 1 == 0:
            torch.save(ckpt, os.path.join(save_path, project + '_epoch_{}.ckpt'.format(epoch+1)))

    # except KeyboardInterrupt:
    #     ckpt = {
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': running_loss,
    #             'auc': auc}
    #     torch.save(ckpt, os.path.join(save_path, 'last.ckpt'))


# python panda.py --settings_path settings\panda.hjson --dataset custom --ewc

def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader, desc='Training')):
        imgs, labels, label_names, filenames = batch

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


def get_score(model, device, train_loader, val_loader):
    train_feature_space = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='Train set feature extracting'):
            # imgs, _ = batch
            # imgs = imgs.to(device)

            imgs, y, _, _ = batch
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    val_feature_space = []
    with torch.no_grad():
        labels = []
        val_accuracy = []
        for batch in tqdm(val_loader, desc='Val set feature extracting'):
            # imgs, _ = batch
            # imgs = imgs.to(device)

            imgs, y, _, _ = batch
            labels.extend(y.type(torch.int).tolist())

            imgs = imgs.to(device)
            yhat, features = model(imgs)
            yhat = yhat.unsqueeze(1)
            accuracy = ClassAccuracy(yhat, y)
            val_accuracy.append(accuracy)


            val_feature_space.append(features)
        val_feature_space = torch.cat(val_feature_space, dim=0).contiguous().cpu().numpy()

    val_accuracy = np.average(np.array(val_accuracy))
    distances = utils.knn_score(train_feature_space, val_feature_space)

    auc = roc_auc_score(labels, distances)

    return auc, train_feature_space

def main(args):

    settings = load_settings(args.settings_path)
    # settings = update_settings(args, settings)

    save_path = os.path.join(args.save_folder, 'ckpt')
    if not os.path.exists(args.save_folder):
        os.makedirs(save_path)

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

    if args.dataset == 'custom':
        dm = create_datamodule(settings, run=None)
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        train_model(model, train_loader, val_loader, device, args, ewc_loss, save_path, args.project)
    else:
        train_loader, val_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)
        train_model(model, train_loader, val_loader, device, args, ewc_loss, save_path, args.project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--settings_path', default='settings/panda.hjson', type=str, help='Path to settings file')
    parser.add_argument('--dataset', choices=['cifar10', 'fashion', 'custom'], default='custom')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_folder', default='training', type=str, help='Output folder to save model checkpoints')
    parser.add_argument('--project', default='training', type=str, help='Project name')

    args = parser.parse_args()

    main(args)
