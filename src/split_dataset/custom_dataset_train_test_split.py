import logging 


from src.loader.custom_dataset import CustomDataset

class CustomDatasetTestSplitter():
    def __init__(self,dset: CustomDataset):
        logging.info("Started the custom dataset splitter")
    

    def __get_all_cats__(self)->list[str]:
        """gets a de-duplicated list of categories 
        from the dataset

        Returns:
            list[str]: de duplicated list of categories
            in the dataset
        """
        all_cat = []
        for label_current in self.dset.labels:
            all_cat.append(label_current.split("_")[1:])
        return list(set(all_cat))


    def __label_split__(self):
        
        all_labels = self.__get_all_cats__()
        for label_current in self.dset.labels:
            logging.debug(f"The current label is {label_current}")
            instance = label_current.split("_")[0]
            category = label_current.split("_")[1]
        #get all the categories 
        #find the number of instances per category
        #split based on percentage 

        # plan _--_--__-_-
        # get the list of all labels 
        # loop through this and do is in to find matching instances
        # in that category, then count the number per category
        # shuffle and split this based on the train test split percentage
        
