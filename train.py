import numpy as np 
import pandas as pd 
import cv2 
import matplotlib.pyplot as plt 
import os 
import shutil 
from siamese_model import SiameseModel
from data_loader import *
from meta_class_gen import * 

data_dir = os.path.join("dataset/pokemon-images-scraped-from-bulbapedia/scraped_cleaned_200/")
classes_inorder = [_ for _ in os.listdir(data_dir)]


def train(): 
    siamese_data_gen = SiameseDataGenerator()
    training_classes, support_classes, query_classes = create_meta_dataset(data_dir, 5)
    train_X, train_y = siamese_data_gen.create_training_data(data_dir, training_classes, classes_inorder)
    num_classes = len(classes_inorder)
    combined_data, meta_y = siamese_data_gen.make_paired_data(train_X, train_y, num_classes)
    support_set, support_y, query_set = siamese_data_gen.create_n_way_data(5, data_dir, support_classes, query_classes, classes_inorder)
    positive_sample = combined_data[:, :6]
    negative_sample = combined_data[:, 5900:5906]
    visulization_data = np.hstack((positive_sample, negative_sample))
    visualization_y = np.concatenate((meta_y[: 6], meta_y[5900: 5906]), axis=0)
    siamese_data_gen.visualize(visulization_data, visualization_y, 12, 3)

    train_1 = combined_data[0, :] / 255.0
    train_2 = combined_data[1, :] / 255.0

    del train_X
    del combined_data
    del train_y
    del positive_sample
    del negative_sample
    del visulization_data

    siamese_model = SiameseModel()
    siamese_model.build()
    history, model = siamese_model.train(train_1, train_2, meta_y, 25, 32)
    siamese_model.save("models/siamese.hd5")

    res = siamese_model.pred_query_set(model, 2, support_set, support_y, query_set)

    return res

if __name__ == "__main__":
    train()


