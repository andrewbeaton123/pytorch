import logging

from src.loader.custom_dataset import CustomDataset

from src.transforms import compose_for_cifar_ten,general_transform,mushroom_transformer
from src.load_cifar_ten import load_train_test_cifar_ten

logging.basicConfig(level="INFO")



def main_cifar():
    logging.info("Starting Main ... ")
    #cifar_compose = compose_for_cifar_ten()
    general_tf = general_transform([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    logging.info("Generated compose options")
    train_cfar, test_cfar = load_train_test_cifar_ten(general_tf)
    logging.info("finished loading CIFAR")
    print(train_cfar)
    print(test_cfar)
    
def main():

    logging.info("Starting main function")
    

    mush_data  = CustomDataset(r"C:\Users\andrewb\Documents\DSets\data\mushrooms\\",mushroom_transformer())
    print(mush_data.__len__())
    
    print(f"Label types are : {mush_data.labels}")
    logging.info("Finished main")
if __name__ == '__main__':
    main()
