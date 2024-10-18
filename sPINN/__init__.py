import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import trange

def dydx(yi, x, n):
        """
        General formula to compute the n-th order derivative of y = f(x) 
        with respect to x.
        
        Attributes:
            yi (torch tensor): the tensor to take the derivative of. Requires grad.
            x (torch tensor): Variable to take the derivative with respect to. Requires grad
            n (int): order of the derivative

        Returns:
            (tensor): n-th derivative of y with respect to x.
        
        """
    
        if n == 0:
            return yi
        else:
            dy_dx = torch.autograd.grad(yi, x, torch.ones_like(yi), create_graph=True, retain_graph=True, allow_unused=True)[0]
        return dydx(dy_dx, x, n - 1)
    
    
class PINN(nn.Module):
    """
    A class used to represent a PINN for solving a differential equation

    Methods:
        __init__(self, hidden_sizes):
            Initializes the PINN with len(hidden_sizes) hidden layers with "hidden_sizes" neurons.
        
            Attributes:
                hidden_sizes (array, int): Array containing the number of neurons in each hidden layer.

        set_trainer(self, optimizer, scheduler, loss):
            Incoportates a defined optimizer, scheduler and loss function into the class.

            Attributes:
                optimizer (function): Optimizer for training the model.
                scheduler (function): Scheduler to control the change of the learning rate. Can be None
                loss (function): Loss function for training the model. It must be of the form
                    loss(y (tensor), x (tensor)) --> tensor(1,)


        set_integration_regime(self, N, xi, xf):
            Initializes the integration region for the resampling.

            Attributes:
                N (int): Number of points in the grid.
                xi (float): Left boundary of the integration regime.
                xf (float): Right boundary of the integration regime.

        forward(self, x):
            Returns the prediction on the set of points x (torch array).

            Attributes:
                x (tensor(N,)): Tensor of n input elements to evaluate. 

            Returns:
                (tensor(N,)): Prediction of the model in the points x.
        
        train(self, x, epochs, 
              target = 1e-9, res_step = 100, resampling = True, schedule = True):
        
            Trains the model using the set of inputs x.
        
            Attributes:
                x (tensor(N,), requires_grad): Initial input points.
                epochs (int): Number of epochs for training the model.
                target (float): Target value for the loss. Training stops if this is reached.
                res_step (int): Number of steps after which the loss value is shown in screen and
                    the x points are resampled (if resampling = True).
                resampling (bool): Sets whether x points are resampled every res_step epochs.
                schedule (bool): Sets whether the scheduler is used to adjust the learning rate.

        plot_loss_history(self):
            Plots the history of values of the loss function during training. It can also be accessed as a
            dictionary by calling PINN.history.
        
    """   
    
    def __init__(self, hidden_sizes):
        super(PINN, self).__init__()
        self.num_layers = len(hidden_sizes)
        self.sizes = hidden_sizes
       
        self.input_layer = nn.Linear(1, self.sizes[0])
        
        self.hidden_layers = [
            nn.Linear(self.sizes[j], self.sizes[j+1]) for j in range(self.num_layers-1)
        ]  # the stack of num_layers hidden layers
        
        self.output_layer = nn.Linear(self.sizes[-1],1)

        self.history = {"loss": []}
        
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        
        self.N = None
        self.xi = None
        self.xf = None
        
    def set_trainer(self, optimizer, scheduler, loss):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        
    def set_integration_regime(self, N, xi, xf):
        self.N = N
        self.xi = xi
        self.xf = xf
        
    def forward(self, x):
        x = torch.reshape(x, (len(x),1)) 
        
        x = self.input_layer(x)
        x = nn.Tanh()(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.Tanh()(x)
   
        x = self.output_layer(x)

        x = x.reshape(len(x))
        
        return x
    
    def train(self, x, epochs, 
              target = 1e-9, res_step = 100, resampling = True, schedule = True):

        hist = []
        learning_rate = self.scheduler.get_last_lr()[-1]


        for step in trange(1, epochs+1):
       
            self.optimizer.zero_grad()
        
            y = self.forward(x)
            loss_value = self.loss(y, x)
            
            if loss_value < target:
                break
        
            loss_value.backward()
            self.optimizer.step()
        
        
            if step%res_step == 0:
            
                print("Epoch: %d. Loss: %.6E. Learning rate: %6E." % (step, loss_value, learning_rate))
        
                if resampling:
                    random_tensor = torch.rand(self.N, dtype= torch.float32)
                    x = self.xf + (self.xi - self.xf) * random_tensor
                    x = x.clone().detach().requires_grad_(True)
                
            if schedule:      
                self.scheduler.step()
                learning_rate = self.scheduler.get_last_lr()[-1]
                    
    
            
            hist.append(loss_value.item())
        
        
        print("-------------------------------------")
        print(
            "Finished after %d epochs. \n Training loss: %.6E"
            % (epochs, loss_value.item())
        )
    
        self.history["loss"] = np.concatenate((self.history["loss"],hist))
        
    def plot_loss_history(self):
        plt.semilogy(self.history["loss"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss value")
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.show()


class BONDI(nn.Module):
    """
    Class used to solve the problem of Bondi accretion. It outputs the values concatenated.

    Methods:
        __init__(self, hidden_sizes):
            Initializes the PINN with len(hidden_sizes) hidden layers with "hidden_sizes" neurons.
        
            Attributes:
                hidden_sizes (array, int): Array containing the number of neurons in each hidden layer.

        set_trainer(self, optimizer, scheduler, loss):
            Incoportates a defined optimizer, scheduler and loss function into the class.

            Attributes:
                optimizer (function): Optimizer for training the model.
                scheduler (function): Scheduler to control the change of the learning rate. Can be None
                loss (function): Loss function for training the model. It must track the previous loss values for using adaptative learning and be of the form (not the most elegant form but it works)
                    loss(y(tensor), x(tensor), old_l1(tensor(1,)), old_l2(tensor(1,))) 
                        --> (tensor(1,),tensor(1,),tensor(1,))


        set_integration_regime(self, N, xi, xf):
            Initializes the integration region for the resampling.

            Attributes:
                N (int): Number of points in the grid.
                xi (float): Left boundary of the integration regime.
                xf (float): Right boundary of the integration regime.

        forward(self, x):
            Returns the prediction on the set of points x (torch array).

            Attributes:
                x (tensor(N,)): Tensor of n input elements to evaluate. 

            Returns:
                (tensor(2,N)): Prediction of the model in the points x.
        
        train(self, x, epochs, 
              target = 1e-9, res_step = 100, resampling = True, schedule = True):
        
            Trains the model using the set of inputs x.
        
            Attributes:
                x (tensor(N,), requires_grad): Initial input points.
                epochs (int): Number of epochs for training the model.
                target (float): Target value for the loss. Training stops if this is reached.
                res_step (int): Number of steps after which the loss value is shown in screen and
                    the x points are resampled (if resampling = True).
                resampling (bool): Sets whether x points are resampled every res_step epochs.
                schedule (bool): Sets whether the scheduler is used to adjust the learning rate.

        plot_loss_history(self):
            Plots the history of values of the loss function during training. It can also be accessed as a
            dictionary by calling PINN.history.
        
    """   
    
    def __init__(self, hidden_sizes):
        super(BONDI, self).__init__()
        self.num_layers = len(hidden_sizes)
        self.sizes = hidden_sizes


        # Network 1
        self.input_layer_1 = nn.Linear(1, self.sizes[0])
        
        self.hidden_layers_1 = [
            nn.Linear(self.sizes[j], self.sizes[j+1]) for j in range(self.num_layers-1)
        ]  # the stack of num_layers hidden layers
        
        self.output_layer_1 = nn.Linear(self.sizes[-1],1)

        # Network 2
        self.input_layer_2 = nn.Linear(1, self.sizes[0])
        
        self.hidden_layers_2 = [
            nn.Linear(self.sizes[j], self.sizes[j+1]) for j in range(self.num_layers-1)
        ]  # the stack of num_layers hidden layers
        
        self.output_layer_2 = nn.Linear(self.sizes[-1],1)
        

        self.history = {"loss": [], "loss_1": [], "loss_2": []}
        
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        
        self.N = 1000
        self.xi = 0
        self.xf = 1
        
    def set_trainer(self, optimizer, scheduler, loss):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        
    def set_integration_regime(self, N, xi, xf):
        self.N = N
        self.xi = xi
        self.xf = xf
        
    def forward(self, x):
        x = torch.reshape(x, (len(x),1)) 

        x1 = self.input_layer_1(x)
        x1 = nn.Tanh()(x1)
        for layer in self.hidden_layers_1:
            x1 = layer(x1)
            x1 = nn.Tanh()(x1)
        x1 = self.output_layer_1(x1)

        x2 = self.input_layer_2(x)
        x2 = nn.Tanh()(x2)
        for layer in self.hidden_layers_2:
            x2 = layer(x2)
            x2 = nn.Tanh()(x2)
        x2 = self.output_layer_2(x2)
        
        x = torch.cat((x1.reshape(1,len(x)),x2.reshape(1,len(x))))
        return x

    
    def train(self, x, epochs, target = 1e-9, res_step = 100, resampling = True, schedule = True):
        
        optimizer = self.optimizer
        scheduler = self.scheduler
        loss = self.loss
        history = {"loss": [], "loss_1":[], "loss_2": []}
        learning_rate = self.scheduler.get_last_lr()[-1]

        print("Training for %d epochs with learning rate %.2E\n" % (epochs, learning_rate))
        print("-------------------------------------")
        l_rate = learning_rate
        old_l1 = torch.tensor(10, dtype = torch.float32)
        old_l2 = torch.tensor(10, dtype = torch.float32)

        for step in trange(1, epochs+1):
       
            optimizer.zero_grad()
        
            y = self.forward(x)
            loss_value, l1, l2 = loss(y, x, old_l1, old_l2)
        
        
            loss_value.backward()
            optimizer.step()
        
            old_l1 = torch.clone(l1).detach()
            old_l2 = torch.clone(l2).detach()

            if loss_value < target:
                break
        
            if step%res_step == 0:
            
                print("Epoch: %d. Loss: %.6E. Learning rate: %6E." % (step, loss_value, learning_rate))
        
                if resampling:
                    random_tensor = torch.rand(self.N, dtype= torch.float32)
                    x = self.xf + (self.xi - self.xf) * random_tensor
                    x = x.clone().detach().requires_grad_(True)
                
            if schedule:
                
                scheduler.step()
                learning_rate = scheduler.get_last_lr()[-1]
        
            history["loss"].append(loss_value.item())
            history["loss_1"].append(old_l1.item())
            history["loss_2"].append(old_l2.item())
            
        

        self.history["loss"] = np.concatenate((self.history["loss"],history["loss"]))
        self.history["loss_1"] = np.concatenate((self.history["loss_1"],history["loss_1"]))
        self.history["loss_2"] = np.concatenate((self.history["loss_2"],history["loss_2"]))
        
        print("-------------------------------------")
        print(
            "Finished after %d epochs. \n Training loss: %.6E"
            % (epochs, loss_value.item())
        )
    

    def plot_loss_history(self):
        history = self.history
        plt.semilogy(history["loss_1"], label = "Eq1 Loss")
        plt.semilogy(history["loss_2"], label = "Eq2 Loss")
        plt.semilogy(history["loss"], label = "Mean Loss")

        plt.xlabel("Epochs")
        plt.ylabel("Loss value")
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()
       
