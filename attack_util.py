import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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

    def _projection(self, a, eps):
        """
        An infinity norm projection basically clips a to eps:
            d_i = {
                -eps if a_i < eps
                a_i if eps <= a_i <= eps
                eps if a_i > eps
            }

        """
        # Note: Following slide 17 from lecture "Lp attack with notes" 
        #  - Note that we're doing an infinity-norm attack, not l1-norm attack.
        #    Therefore, we can't directly use the slides (but something close).
        return torch.clamp(a, min=-eps, max=eps)

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
            delta_hat = delta - self._alpha * torch.sign(delta.grad)
            delta.grad.zero_()      # Clear gradient for the next pass

            # Sanity check to make sure gradient is actually getting computed properly.
            # Note that this is a lower bound on the loss, since delta_hat may not be
            # within the epsilon-ball constraints
            # Note: Commented for speed; please uncomment when testing new changes
            # assert(self.ce_loss(model(X + delta_hat), y) < loss)

            # Calculate projection of delta onto epsilon ball
            delta = self._projection(delta_hat, self._eps)
            delta = torch.tensor(delta, requires_grad=True)

            # Sanity check to make sure that the new perturbation will still
            # lower the attack loss
            # Note: Commented for speed; please uncomment when testing new changes
            # assert(self.ce_loss(model(X + delta), y) < loss)

            # An alternate way to check the infinity-norm constraint
            assert(torch.max(torch.abs(delta)) <= self._eps)
        
        return delta


### FGSMAttack
'''
Technically you can transform your PGDAttack to FGSM Attack by controling parameters like `attack_step`. 
If you do that, you do not need to implement FGSM in this class.
'''
class FGSMAttack():
    def __init__(self, eps=8 / 255, loss_type='ce', targeted=True, num_classes=10):
        # FGSM is just a special case of PGD
        self._pgd_attack = PGDAttack(attack_step=1, eps=eps, loss_type=loss_type,
                                     num_classes=num_classes, targeted=targeted,
                                     alpha=eps)

    def perturb(self, model: nn.Module, X, y):
        return self._pgd_attack.perturb(model, X, y)



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
