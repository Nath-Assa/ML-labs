import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

def train_val_split(x, y, val_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    val_split_idx = int(x.shape[0] * val_size)
    train_indices, val_indices = indices[val_split_idx:], indices[:val_split_idx]
    return x[train_indices], x[val_indices], y[train_indices], y[val_indices]

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    
    if args.nn_type in ["transformer", "cnn"]:
        image_size = int(np.sqrt(xtrain.shape[1]))
        xtrain = xtrain.reshape(xtrain.shape[0], 1, image_size, image_size)
        xtest = xtest.reshape(xtest.shape[0], 1, image_size, image_size)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
        
        
    means = np.mean(xtrain, axis=0)
    stds = np.std(xtrain, axis=0)
    
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)

    # Make a validation set
    if not args.test:
        xtrain, xval, ytrain, yval = train_val_split(xtrain, ytrain, val_size=0.2)
    else:
        xval, yval = None, None  # No validation set if evaluating on test data

    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        if not args.test:
            xval = pca_obj.reduce_dimension(xval)
        xtest = pca_obj.reduce_dimension(xtest)


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
       model = MLP(input_size=xtrain.shape[1], n_classes=n_classes)
    elif args.nn_type == "cnn":
        model = CNN(input_channels=1, n_classes=n_classes)
    elif args.nn_type == "transformer":
        model =  MyViT(chw=(1, 28, 28), n_patches=7, n_blocks=6, hidden_d=64, n_heads=8, out_d=n_classes)
    else:
        raise ValueError("Invalid nn_type")

    model.to(args.device)
    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=args.device)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)
    
    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    
    # Predict on unseen data
    if  args.test:
        preds = method_obj.predict(xtest)
        np.save("predictions", preds)
        print("Test set predictions saved. Submit this file to AIcrowd for evaluation.")
    else:
        preds = method_obj.predict(xval)
        y_true = yval
        # Report results: performance on validation set
        acc_val = accuracy_fn(preds, y_true)
        macrof1_val = macrof1_fn(preds, y_true)
        print(f"Validation set: accuracy = {acc_val:.3f}% - F1-score = {macrof1_val:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
