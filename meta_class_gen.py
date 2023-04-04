import numpy as np 
import random

import random

def create_meta_dataset(data_path, k_way):
    assert type(k_way) == int, "k_way: requires integer datatype."
    classes = os.listdir(data_path)
    support_classes = random.choices(classes, k=k_way)
    train_classes = [c for c in classes if not c in support_classes]
    query_classes = support_classes

    return train_classes, support_classes, query_classes
