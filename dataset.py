import torch
import torchvision
import torchvision.transforms as transforms


def return_data(args):
    """docstring for data_loader"""
    image_size = args.image_size  # images are 64 * 64 * 1
    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # print("input data: ",args.dataset)
    if args.dataset == 'MNIST' or args.dataset.lower() == 'mnist':
        print('Load MNIST data now....')
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=4)

    elif args.dataset == 'SVHN' or args.dataset.lower() == 'svhn':
        print('Load SVHN data now....')
        trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                             download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
        testset = torchvision.datasets.SVHN(root='./data', split='test',
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=4)

    elif args.dataset == 'CIFAR10' or args.dataset.lower() == 'cifar10':
        print('Load CIFAR10 data now....')
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=4)
    else:
        raise ('please choose the Right Dataset')

    if args.train:
        return trainloader
    else:
        print('-' * 80)
        print("loading testing data now-")
        print('-' * 80)
        return testloader

