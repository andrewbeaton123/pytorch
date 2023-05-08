import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_train_test_cifar_ten(compose_object:transforms.Compose):
    
    # Load the CIFAR-10 training and test sets
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=compose_object)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=compose_object)
    return trainset,testset

