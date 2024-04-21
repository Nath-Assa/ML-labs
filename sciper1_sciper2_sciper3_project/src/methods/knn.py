import numpy as np

class KNN:
    """
    k-Nearest Neighbors classifier object.
    """
    
    def __init__(self, k=100, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels

        # Predict on the training data using the predict method
        pred_labels = self.predict(self.training_data)
        return pred_labels  
        
    def predict(self, test_data):

        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        N = len(test_data) # Lenght of the test data set 
        
        if(self.task_kind == "classification") :
            test_labels = np.zeros(N) # This array will hold the predicted labels for each test sample
            
            # Predict on the test data 
            test_labels = self.knn(test_data,self.k)  
        else :
            test_labels = np.zeros((N,2)) # This array will hold the predicted labels, shape (N,2) appropriate for regression task 
            for i in range(N) : 
                distances = self.compute_distances(test_data[i]) # Compute the distances between sample i, and all training data samples  
                idxs = self.k_nearest_neighbors(distances) # Indices of the k nearest neighbors of sample i 
                n_distances = distances[idxs] # Distances of the k nearest neighbors 
                n_ys = self.training_labels[idxs] # Labels of the k nearest neighbors 
                if n_distances[0] == 0 : 
                    test_labels[i] = np.mean(n_ys) 
                else :
                    weights = 1 / n_distances
                    test_labels[i] = np.sum(weights[:, np.newaxis] * n_ys, axis=0) / np.sum(weights # Average value of the K nearest neighbors values 

        return test_labels


    def compute_distances(self,exemple): # Compute the Euclidean distance between a single example vector, and all training samples 
        values = np.square(self.training_data - exemple)
        distances = np.sqrt(np.sum(values,axis=1))
        return distances # Return the distances to all training samples

    def k_nearest_neighbors(self,distances): # Find the k nearest neighbors
        indices = np.argsort(distances)
        return indices[:self.k] # Return the indices of the k nearest neighbors

    def predict_label(self,neighbors_labels): # Assign to the example vector, the most common label over its k nearest neighbors 
        nb_occurences = np.bincount(neighbors_labels)
        return np.argmax(nb_occurences) # Return the predict label 
    
    def knn_single_sample(self,single_sample) : # Apply all required steps to find the predicted label of a single sample 
        distances = self.compute_distances(single_sample) 
        idxs = self.k_nearest_neighbors(distances)
        n_labels = self.training_labels[idxs]
        single_sample_prediction = self.predict_label(n_labels)
        return single_sample_prediction
        
    def knn(self,samples,k) : # Apply the knn_single_sample function to an arbitraty number of samples 
        return np.apply_along_axis(self.knn_single_sample, axis= 1, arr=samples)
