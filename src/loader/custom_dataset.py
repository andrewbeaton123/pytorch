import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging 

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        logging.info("Custom dataset created")
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Get the list of image paths and corresponding labels
        self._load_dataset()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_dataset(self):
        classes = os.listdir(self.root_dir)
        classes.sort()  # Sort classes alphabetically for consistent label assignment
        logging.info("Starting dataset loading")
        logging.debug(f"The classes are ...{classes}")

        for i, class_name in enumerate(classes):
            logging.debug(f"i = {i}")
            logging.debug(f"Class name is {class_name}")
            class_dir = os.path.join(self.root_dir, class_name)
            
            if os.path.isdir(class_dir):

                image_names = os.listdir(class_dir)
                image_paths = [os.path.join(class_dir, img_name) for img_name in image_names]
                #logging.debug(f"Image paths are {image_paths}")
                self.image_paths.extend(image_paths)
                logging.debug(f"The length of the paths is {len(image_paths)}")
                
                clean_names = [str(number.split(".")[0])+"_"+class_name for number in image_names]
                
                self.labels.extend(clean_names)
                #self.labels.extend([i_name] * len(image_paths))
                logging.debug(f"Labels are  {self.labels}")
        logging.info("Finished loading dataset")
        #logging.debug(f"labels are {self.labels}")