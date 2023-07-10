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
def mushroom_transformer() -> transforms.Compose :
    """holds the transformer to handle the mushroom data


    Returns:
        transforms.Compose: Resized and scalled tensor transformer
    """
    return  transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to a specific size
    transforms.ToTensor(),  # Converts the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])
