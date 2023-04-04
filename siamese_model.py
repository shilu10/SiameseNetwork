from tensorflow.keras.layers import Dense, Input, Lambda, Add, Conv2D, MaxPooling2D, BatchNormalization, Flatten, AveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
import tensorflow as tf 
import numpy as np 
import random 
import os 


class SiameseModel: 
    
    def __init__(self): 
        self.siamese_model = None
    
    def EmbeddingConvLayer(self):
        input =Input((100, 100, 3))
        x = BatchNormalization()(input)
        x = Conv2D(32, (5, 5), activation="tanh")(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Conv2D(16, (5, 5), activation="tanh")(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)

        x = BatchNormalization()(x)
        x = Dense(10, activation="tanh")(x)
        embedding_network = keras.Model(input, x)

        return embedding_network 
    
    def euclidean_distance(self, vects):
        """
        Find the Euclidean distance between two vectors.

        Arguments:
            vects: List containing two tensors of same length.

        Returns:
            Tensor containing euclidean distance
            (as floating point value) between vectors.
        """
        x, y = vects
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

    def loss(self, margin=1):
        def contrastive_loss(y_true, y_pred):
            """
            Calculates the constrastive loss.

            Arguments:
                y_true: List of labels, each label is of type float32.
                y_pred: List of predictions of same length as of y_true,
                        each label is of type float32.

            Returns:
                A tensor containing constrastive loss as floating point value.
            """
            square_pred = tf.math.square(y_pred)
            margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
            return tf.math.reduce_mean(
                (1 - y_true) * square_pred + (y_true) * margin_square
            )

        return contrastive_loss
    
    def build(self): 
        input_1 = Input((100, 100, 3))
        input_2 = Input((100, 100, 3))

        embedding_network = self.EmbeddingConvLayer()
        net1 = embedding_network(input_1)
        net2 = embedding_network(input_2)
        merge = Lambda(self.euclidean_distance)([net1, net2])
        normal = BatchNormalization()(merge)
        _output = Dense(1, activation="sigmoid")(normal)
        siamese_model = keras.Model(inputs=[input_1, input_2], outputs=_output)
        siamese_model.compile(optimizer=Adam(learning_rate=0.0001), 
                                          loss=self.loss(margin=1), metrics="acc")
        
        self.siamese_model =  siamese_model
    
    def train(self, input1, input2, y, epochs, batch_size):
        history = self.siamese_model.fit([input1, input2], y, batch_size=batch_size,
                                                     epochs=epochs)
        return history, self.siamese_model
    
    def pred_query_set(self, model, n_way, support_set, support_y, query_set): 
    
        result = []
        for query_img in query_set: 
            pred = [0 for _ in range(len(support_set))]
            plt.imshow(query_img)
            print("query_img")
            plt.show()
            print()
            for ind, inner_support_set in enumerate(support_set):
                for i in range(n_way): 
                    if len(inner_support_set) >= i+1: 
                        sup_img = inner_support_set[i]
                        plt.imshow(sup_img)
                        plt.show()
                        print('supo')
                        sup_img = np.reshape(sup_img, (1, 100, 100, 3))
                        query_img = np.reshape(query_img, (1, 100, 100, 3))
                        prob_val = model.predict([sup_img, query_img], verbose=False)
                        print(prob_val, ind, "prediction")
                        pred[ind] += prob_val
                pred[ind] += pred[ind] / len(inner_support_set)
            result.append(np.array(pred).argmax())
            print(result)
            
        return result
        
    def save(self, file_path): 
        self.siamese_model.save(file_path)
        print("Saved Model Successfully")
