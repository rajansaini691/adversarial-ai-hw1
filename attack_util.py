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
        ### Your code here
        pass
        ### Your code ends

    def ce_loss(self, logits, y):
        """
        Args:
            logits  Tensor of dim (64, 10)      <--- (BATCH_SIZE, NUM_CLASSES)
            y       Tensor of dim (64)
        """
        log_logits = torch.log(logits)
        one_hot_labels = torch.nn.functional.one_hot(y, num_classes=10).float()
        return torch.sum(torch.mul(log_logits, one_hot_labels), dim=1)

    def cw_loss(self, logits, y):
        """
        Args:
            logits  Tensor of dim (64, 10)      <--- (BATCH_SIZE, NUM_CLASSES)
            y       Tensor of dim (64)
        """
        ### Your code here
        pass
        ### Your code ends

    def _bisection_algorithm(func, a, b):
        """
        Returns x such that func(x) = 0 using the bisection method

        Parameters:
            func        The function whose root we are looking for. It should accept a float
                        and return a float (or accept a vector and return a vector of the
                        same dimension, depending on whether we want to parallelize)
            a           Lower bound for x (func(a) should be negative)
            b           Upper bound for x (func(b) should be positive)
        """
        assert(func(a) < 0)         # TODO This should be true element-wise for vectors
        assert(func(b) > 0)         # TODO This should be true element-wise for vectors

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
        delta = torch.zeros_like(X)
        print(y)
        
        # TODO Put this in a loop iterating for attack_step steps
            # TODO Calculate gradient of ce_loss w.r.t. delta using autograd. Model should
            #      take x + delta as input
            #      Then use update rule to create delta_hat = delta_k - alpha * delta_grad
            #      For sanity checking (check gradient), assert that
            #          attack_loss(model(x + delta_hat)) < attack_loss(model(x + delta_k))
            #      Then calculate projection of delta onto epsilon ball
        
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
