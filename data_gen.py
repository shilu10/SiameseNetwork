import random 
import tensorflow as tf
import numpy as np 
import time
import shutil
import os
import tensorflow.keras as tf 
import matplotlib.pyploy as plt

class SiameseDataGenerator: 
    
    def __init__(self):
        pass 
    
    def create_training_data(self, train_data_path, training_classes, classes_inorder): 
        train_X = []
        train_y = []

        for c in training_classes: 
            label_index = classes_inorder.index(c)
            files = next(os.walk(train_data_path + "/" + c))
            files = files[2]
            for f in files:
                file = train_data_path + "/" + c + "/" + f
                img = cv2.imread(file)
                img = cv2.resize(img, (100, 100))
                train_X.append(img)
                train_y.append(label_index)
        return np.array(train_X), np.array(train_y).astype("int")

    def create_support_data(self, support_data_path, support_classes, classes_inorder): 
        support_X = []
        support_y = []

        for c in support_classes: 
            label_index = classes_inorder.index(c)
            files = next(os.walk(support_data_path + "/" + c))
            files = files[2]
            for f in files:
                file = support_data_path + "/" + c + "/" + f
                img = cv2.imread(file)
                img = cv2.resize(img, (100, 100))
                support_X.append(img)
                support_y.append(label_index)
        return np.array(support_X), np.array(support_y).astype("int")

    def create_query_data(self, query_data_path, query_classes, classes_inorder): 
        query_X = []
        query_y = []

        for c in training_classes[:10]: 
            label_index = classes_inorder.index(c)
            files = next(os.walk(query_data_path + "/" + c))
            files = files[2]
            for f in files:
                file = query_data_path + "/" + c + "/" + f
                img = cv2.imread(file)
                img = cv2.resize(img, (100, 100))
                query_X.append(img)
                query_y.append(label_index)
        return np.array(query_X), np.array(query_y).astype("int")
    
    def select_similar_data(self, y, y_label, curr_index):
        similar_indexes = np.where(y==y_label)
        # print(similar_indexes, "sim index")
        similar_indexes = np.delete(similar_indexes, np.where(similar_indexes==curr_index))
        # print("done")
        selected_index = np.random.choice(similar_indexes)
        return selected_index
    
    def select_dissimilar_data(self, num_classes, y, y_label):
        all_classes = np.arange(num_classes+1)
        all_classes = np.delete(all_classes, np.where(y_label==all_classes)[0][0])

        while True: 
            selected_class = np.random.choice(all_classes)
            if  selected_class in y:
                break 
        disimilar_class_indexes = np.where(y==selected_class)
        
        selected_index = np.random.choice(disimilar_class_indexes[0])
        return selected_index
    
    def make_paired_data(self, X, y, num_classes): 
        similar_data = []
        disimilar_data = []
        meta_y = np.zeros(len(X)*2)
        for i in range(len(X)): 
            similar_index = self.select_similar_data(y, y[i], i)
            similar_data.append(X[similar_index])
            dissimilar_index = self.select_dissimilar_data(num_classes, y, y[i])
            disimilar_data.append(X[dissimilar_index])
            
        similar_data = [train_X, similar_data]
        disimilar_data = [train_X, disimilar_data]
        combined_data = np.hstack((similar_data, disimilar_data))
       
        meta_y[: len(train_X)+1] = 0
        meta_y[len(train_X): ] = 1
        return combined_data, meta_y
    
    def visualize(self, pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
        
        num_row = to_show // num_col if to_show // num_col != 0 else 1
        to_show = num_row * num_col
        # Plot the images
        fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
        for i in range(to_show):
            # If the number of rows is 1, the axes array is one-dimensional
            if num_row == 1:
                ax = axes[i % num_col]
            else:
                ax = axes[i // num_col, i % num_col]
            ax.imshow(tf.concat([pairs[0][i], pairs[1][i]], axis=1), cmap="gray")
            ax.set_axis_off()
            if test:
                ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
            else:
                ax.set_title("Label: {}".format(labels[i]))
        if test:
            plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
        else:
            plt.tight_layout(rect=(0, 0, 1.5, 1.5))
        plt.show()
        
    def create_n_way_data(self, n_way, data_path, support_classes, query_classes, classes_inorder): 
        query_set = []
        support_set = [[] for _ in range(n_way)]
        support_y = []
        for outer_ind, sup_class in enumerate(support_classes): 
            src = data_path + "/" + sup_class
            files = next(os.walk(src))
            for ind, file in enumerate(files[2]): 
                file_src = src + "/" + file
                img = cv2.imread(file_src)
                img = cv2.resize(img, (100, 100))
                if ind == 0: 
                    query_set.append(img)
                else: 
                    support_set[outer_ind].append(img)
                
            support_y.append(classes_inorder.index(sup_class))
        
        return np.array(support_set), np.array(support_y), np.array(query_set)


