import argparse
import copy
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
import torchvision

import dataset
from model import Net, onnx_export




def save_pickle(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)


def data_loader(args):

    path_current_dir = pathlib.Path(__file__).parent
    path_train_images = path_current_dir.joinpath("train_images")

    with open(path_train_images.joinpath("stats.pickle"), mode="rb") as f:
        stats = pickle.load(f)


    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.Normalize((stats["mean"]), (stats["std"]))  # Standardization (Mean, Standard deviations)
        ])

    images = dataset.ShogiPieceDataset(path_train_images, transform)

    save_pickle(images.class_to_idx, path_current_dir.joinpath("models", "classes.pickle"))

    n_images = len(images)
    train_size = int( n_images * 0.8 )
    val_size = n_images - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(images, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)

    print("n_images: {} -> train:{}, validate:{}".format(n_images, train_size, val_size))
    return train_loader, val_loader, images.class_to_idx


def args_():
    parser = argparse.ArgumentParser(description='Shogi piece detection')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args


def transforms_augmentation(mean:float=0.0, std:float=1.0):
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(degrees=(20), fill=(0.0-mean)/std),
            torchvision.transforms.RandomErasing()
        ])
    return transforms


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = transforms_augmentation()(data)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {:3d} [{:6d} / {:6d} ({:3.0f}%)]   Loss: {:.6f}'.format(
                    epoch, i * len(data),
                    len(train_loader.dataset),
                    100 * i / len(train_loader),
                    loss.item())
                )
    

def validate(model, device, test_loader, create_cm=False):
    model.eval()
    loss = 0
    correct = 0

    predicted_class = torch.empty(0, dtype=torch.int64).to(device)  # confusion matrix
    true_class = torch.empty(0, dtype=torch.int64).to(device)  # confusion matrix

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if create_cm:
                predicted_class = torch.cat([predicted_class, torch.argmax(output, dim=1)], dim=0)
                true_class = torch.cat([true_class, target], dim=0)  # torch.int64

    loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print('\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss,
        correct,
        len(test_loader.dataset),
        100. * accuracy))

    return accuracy, predicted_class.to('cpu'), true_class.to('cpu')


def training():
    '''
    Main function
    '''
    # Training settings
    args = args_()

    # Make models directory
    path_current_dir = pathlib.Path(__file__).parent
    path_models_dir = path_current_dir.joinpath("models")
    path_models_dir.mkdir(exist_ok=True)

    torch.manual_seed(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes_ids = data_loader(args)
    num_classes = len(classes_ids)

    model = Net(num_classes).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_accuracy = 0.0
    best_model = None
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuracy, _, _ = validate(model, device, val_loader)
        scheduler.step()
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(model)

    # Save model
    torch.save(best_model.state_dict(), path_models_dir.joinpath("piece.pt"))
    onnx_export(best_model, device, path_models_dir.joinpath("piece.onnx")) 

    # Confusion matrix
    accuracy, predicted_class, true_class = validate(model, device, val_loader, create_cm=True)
    cm = create_cm(num_classes, predicted_class, true_class)
    plot_confusion_matrix( cm, list(classes_ids) )



def plot_history(history):
    '''
    Plot loss, accuracy
    '''
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    # Loss
    ax1.set_title("Loss")
    ax1.plot(history["epoch"], history["train_loss"], label="train")
    ax1.plot(history["epoch"], history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Accuracy
    ax2.set_title("Accuracy")
    ax2.plot(history["epoch"], history["train_accuracy"], label="train")
    ax2.plot(history["epoch"], history["val_accuracy"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.show()



def create_cm(num_classes, predicted_class, true_class):
    '''
    Confusion matrix for multi-class classification
    '''
    # print(predicted_class)
    # print(true_class)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range( true_class.size(0) ):
        p = predicted_class[i].item()
        t = true_class[i].item()
        cm[t, p] += 1

    return cm


def plot_confusion_matrix(cm, classes):
    plt.matshow(cm, fignum=False, cmap="Blues", vmin=0)
    num_classes = len(classes)
    plt.xticks(np.arange(num_classes), classes)
    plt.yticks(np.arange(num_classes), classes)

    plt.colorbar()
    plt.grid(False)

    plt.title("Confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    # plt.gca().xaxis.set_ticks_position('bottom')

    path_current_dir = pathlib.Path(__file__).parent
    png_path = pathlib.Path(path_current_dir.joinpath("models", "confusion_matrix.png"))
    plt.savefig(png_path)

    # print(cm)



if __name__ == '__main__':
    training()
