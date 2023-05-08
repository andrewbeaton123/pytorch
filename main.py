import logging

from src.transforms import compose_for_cifar_ten
from src.load_cifar_ten import load_train_test_cifar_ten
logging.basicConfig(level="INFO")


def main():
    logging.info("Starting Main ... ")
    cifar_compose = compose_for_cifar_ten()
    logging.info("Generated compose options")
    train_cfar, test_cfar = load_train_test_cifar_ten(cifar_compose)
    logging.info("finished loading CIFAR")
    print(train_cfar)
    print(test_cfar)
    
if __name__ == '__main__':
    main()
