import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

### Do not modif the following codes
class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False
        
def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}

def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False

def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]
### Ends


### PGD Attack
class PGDAttack():
    def __init__(self, attack_step=10, eps=8 / 255, alpha=2 / 255, loss_type='ce', targeted=True, 
                 num_classes=10):
        '''
        attack_step: number of PGD iterations
        eps: attack budget
        alpha: PGD attack step size
        '''
        self._attack_step = attack_step
        self._eps = eps
        self._alpha = alpha
        self._loss_type = loss_type
        self._targeted = targeted

    def ce_loss(self, logits, y):
        """
        Args:
            logits  Tensor of dim (64, 10)      <--- (BATCH_SIZE, NUM_CLASSES)
            y       Tensor of dim (64)

        Returns:
            A single scalar representing the attack loss for the given pair
            of logits and labels
        """
        batch_size, num_classes = logits.shape

        # According to the pytorch docs, LogSoftmax has better numerical
        # properties than separately applying softmax and log
        log_softmax_func = torch.nn.LogSoftmax(dim=1)
        log_probabilities = log_softmax_func(logits)

        # Cross entropy requires one-hot labels (y is categorical)
        one_hot_labels = torch.nn.functional.one_hot(
            y, num_classes=num_classes).float()

        # Get element-wise losses (computing dot products between corresponding
        # labels & log-logits in parallel)
        elementwise_losses = torch.sum(torch.mul(log_probabilities, one_hot_labels), dim=1)

        # Get average loss
        return torch.sum(elementwise_losses, dim=0) / batch_size


    def cw_loss(self, logits, y):
        """
        Args:
            logits  Tensor of dim (64, 10)      <--- (BATCH_SIZE, NUM_CLASSES)
            y       Tensor of dim (64)
        """
        ### Your code here
        pass
        ### Your code ends

    def _bisection_algorithm(self, func, a, b, epsilon):
        """
        Returns x such that func(x) = 0 using the bisection method

        Parameters:
            func        The function whose root we are looking for. It should
                        accept a float and return a float
            a           Lower bound for x
            b           Upper bound for x
            epsilon     Terminates when |a - b| < epsilon

        Precondition:
            func(a) * func(b) <= 0 (implies that they have opposite signs)
        """
        assert(func(a) * func(b) <= 0)
        t = (a + b) / 2

        while abs(a-b) >= epsilon or func(t) != 0:
            # Fancy way of checking if sign(f(a)) == sign(f(t))
            if func(a) * func(t) > 0:
                a = t
            else:
                b = t
            t = (a + b) / 2

        return t

    def _projection(self, a, epsilon):
        """
        Let a = delta_hat
        If |a|_1 <= epsilon, output a
        Otherwise, follow the following algorithm::
            Calculate the root of sum(max(|a_i|-\mu/2, 0)) - \eps = 0 using bisection_algorithm above
            Using the calculated mu, return sign(a) * max(|a| - \mu/2, 0) <-- (max, sign, |.| are elementwise)
        """
        pass

    def perturb(self, model: nn.Module, X, y):
        """
        Calculate the perturbation based on the model
        
        Args:
            model
            X           Tensor of dim (64, 3, 32, 32)   <----- (BATCH_SIZE, NUM_CHANNELS, H, W)
            y           Tensor of dim (64)  <---- (BATCH_SIZE)
                         - These are the class labels (0-9)
        """
        # Initialize perturbation. Setting requires_grad to True lets us
        # take the gradient of the attack loss w.r.t. delta.
        delta = torch.zeros_like(X, requires_grad=True)
        
        for it in range(self._attack_step):
            # Compute attack loss and get gradient
            loss = self.ce_loss(model(X + delta), y)
            loss.backward()

            # Update rule
            delta_hat = delta - self._alpha * delta.grad
            delta.grad.zero_()      # Clear gradient for the next pass

            # Sanity check to make sure gradient is actually getting computed properly.
            # Note that this is a lower bound on the loss, since delta_hat may not be
            # within the epsilon-ball constraints
            assert(self.ce_loss(model(X + delta_hat), y) <= loss)

            # TODO Calculate projection of delta onto epsilon ball
        
        return delta


### FGSMAttack
'''
Technically you can transform your PGDAttack to FGSM Attack by controling parameters like `attack_step`. 
If you do that, you do not need to implement FGSM in this class.
'''
class FGSMAttack():
    def __init__(self, eps=8 / 255, loss_type='ce', targeted=True, num_classes=10):
        pass

    def perturb(self, model: nn.Module, X, y):
        delta = torch.ones_like(X)
        ### Your code here

        ### Your code ends
        return delta



""""
1^T = [1,1,1...]
X = [x_1, x_2, ...]

1^T*X = x_1 + x_2 + ...

f(mu) = max(|a_1| - mu/2, 0) + max(|a_2| - mu/2, 0) + ... + max(|a_N| - mu/N, 0) - \eps

Goal: We want mu s.t. f(mu) = 0


mu_a = 0.000001         ==> f(mu) > 0
[must be here]
mu_c = 500              ==> f(mu) < 0

mu_b = 1000             ==> f(mu) < 0


"""
