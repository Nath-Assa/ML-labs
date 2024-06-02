import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_size=128, dropout_rate=0.5, num_layers=3):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):  # -2 for input and output layers
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.fc_out = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.fc_in(x))
        x = self.dropout(x)
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        preds = self.fc_out(x)
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, initial_hidden_channels=16, target_hidden_channels=64, dropout_rate=0.5, num_layers=3):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        super(CNN, self).__init__()

        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        step_size = (target_hidden_channels - initial_hidden_channels) // (num_layers - 1)

        # initial convolution layer
        hidden_channels = initial_hidden_channels
        self.conv_layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        self.bn_layers.append(nn.BatchNorm2d(hidden_channels))

        # additional convolution layers with increasing number of channels
        for _ in range(num_layers - 1):
            next_hidden_channels = hidden_channels + step_size
            self.conv_layers.append(nn.Conv2d(hidden_channels, next_hidden_channels, kernel_size=3, stride=1, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(next_hidden_channels))
            hidden_channels = next_hidden_channels

        self.dropout = nn.Dropout(dropout_rate)

        #(28x28 for FashionMNIST)
        self.flattened_size = hidden_channels * 4 * 4  
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        for i in range(self.num_layers):
            x = F.relu(self.bn_layers[i](self.conv_layers[i](x)))
            x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  
        x = self.dropout(F.relu(self.fc1(x)))  
        preds = self.fc2(x)
        return preds
        
class MyMSA(nn.Module) :
    def __init__(self,D,nb_heads) :
        super(MyMSA,self).__init__()

        self.dimension = D
        self.nb_heads = nb_heads

        assert D % nb_heads == 0 
        d_head = int(D/nb_heads)
        
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.nb_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.nb_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.nb_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,elements) :
        results = []
        for element in elements : 
            seq = [] 
            for head in range(self.nb_heads) :
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                tmp = element[:,head*self.d_head: (head + 1) * self.d_head] 
                q,k,v = q_mapping(tmp), k_mapping(tmp), v_mapping(tmp)

                attention = self.softmax(q @ k.T / ( self.nb_heads ** 0.5))
                seq.append(attention @ v )

            results.append(torch.hstack(seq))
        return torch.stack(results) # before: torch.cat([torch.unsqueeze(r,dim = 0) for r in results])

class MyViTBlock(nn.Module) :
    def __init__(self,hidden_d,nb_heads,mlp_ratio) :
        super(MyViTBlock,self).__init__()
        
        self.hidden_d = hidden_d
        self.nb_heads = nb_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.multi_head_self_attention = MyMSA(hidden_d,nb_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(nn.Linear(hidden_d,mlp_ratio * hidden_d),nn.GELU(),nn.Linear(mlp_ratio * hidden_d,hidden_d))
    
    def forward(self,x) :
        output = x + self.multi_head_self_attention(self.norm1(x))
        output = output + self.mlp(self.norm2(x))
        return output

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super(MyViT,self).__init__()

        #Defining attributes
        self.chw = chw # (C, H, W)
        self.nb_patches = n_patches
        self.nb_blocks = n_blocks
        self.hidden_d = hidden_d
        self.nb_heads = n_heads
        
        # Patches sizes
        assert chw[1] % self.nb_patches == 0 
        assert chw[2] % self.nb_patches == 0 
        self.patch_size = (chw[1] // self.nb_patches, chw[2] // self.nb_patches )  # We've made the assumption that we have a square picture

        # 1. Linear mapper 
        self.input_d = chw[0] * self.patch_size[0] * self.patch_size[1]
        self.linear_mapper = nn.Linear(self.input_d,self.hidden_d)

        # 2. Learnable classification token 
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3. Positional embedding 
        self.positional_embeddings = self.create_positional_embeddings(self.nb_patches ** 2 + 1, hidden_d)
        
        # 4. Transformer encoder blocks 
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, self.nb_heads) for _ in range(self.nb_blocks)])
        
        # 5. Classification MLP 
        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=1))

    def patchify(self, x):
        n, c, h, w = x.shape

        assert h == w
        nb_patches = self.nb_patches
        patch_size = h // nb_patches

        patches = torch.zeros(n, nb_patches**2, c * patch_size * patch_size)
    
        for index, image in enumerate(x) :
            for i in range(nb_patches):
                for j in range(nb_patches) :
                    patch = image [:, i * patch_size : (i + 1) * patch_size, j * patch_size :(j + 1) * patch_size]
                    patches[index, i * nb_patches + j] = patch.flatten() 

        return patches
    
    def create_positional_embeddings(self, length, D):
        result = torch.ones(length, D)
        for i in range(length):
            for j in range(D):
                result[i][j] = np.sin(i / (10000 ** (j / D))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / D)))
        return result
     

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # First have to patch the data 
        N, C, H, W = x.shape

        # Divide image into patches 
        patches = self.patchify(x)

        # Map the vector corresponding to each patch to the hidden size dimension 
        tokens = self.linear_mapper(patches)

        # Add classification tokens 
        tokens = torch.cat((self.class_token.expand(N, 1, -1), tokens), dim=1)

        # Adding positional embedding 
        preds = tokens + self.positional_embeddings

        # Transformer blocks 
        for block in self.blocks : 
            preds = block(preds)

        # Get classification token 
        preds = preds[:,0]
        
        # Map to the output distribution 
        preds = self.mlp(preds)

        return preds

class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = ...  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
        
