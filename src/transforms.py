import pandas as pd

import torchvision.transforms as transforms
#.transforms as transforms


def compose_for_cifar_ten() -> transforms.Compose:
     """
     returns composes transform which changesdata to tensor 
     and sets the normalized mean and standard dev for the CIFAR-10 
     dataset

     """
     return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
          mean = [0.485, 0.456, 0.406],
          std = [0.229, 0.224, 0.225]
         )]
     )

def general_transform(mean_val, std_val) -> transforms.Compose : 
    """ general transformer that takes in mean 
    and std values as lists of f"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean_val, std_val)
        ]

    )
