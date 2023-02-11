"""
Tests to make sure that the sgd update step is correct
"""
from attack_util import PGDAttack
import torch

def loss_test_helper(num_test_classes, batch_size, loss_correct_pred_threshold,
                     loss_incorrect_pred_threshold, loss_fn, logit_variance,
                     logit_mean):
    """
    Args:
        num_test_classes                Number of classes in dummy input
        batch_size                      Batch size of dummy input
        loss_correct_pred_threshold     For correct predictions, loss should be
                                        above this val
        loss_incorrect_pred_threshold   For incorrect predictions, loss should
                                        be below this val
        loss_fn                         The loss function being tested
        logit_variance                  In the dummy logits, specifies the "spread"
                                        of the values
        logit_mean                      In the dummy logits, specifies the "center"
                                        of the values
    """
    dummy_labels_correct = torch.arange(0, batch_size) % num_test_classes
    dummy_labels_incorrect = torch.cat(
        (torch.arange(1, batch_size), torch.tensor([0]))) % num_test_classes

    # Logits should match the correct labels. The corresponding
    # logit should be high for the correct class and low for all
    # others (e.g. if the class is 3, then logits could be
    # [-25, -25, -25, 25, -25, ...])
    dummy_logits = torch.nn.functional.one_hot(
        dummy_labels_correct, num_classes=num_test_classes)
    dummy_logits = (2 * dummy_logits - 1.) * logit_variance + logit_mean

    correct_label_loss = loss_fn(dummy_logits, dummy_labels_correct)
    incorrect_label_loss = loss_fn(dummy_logits, dummy_labels_incorrect)

    assert(correct_label_loss >= loss_correct_pred_threshold)
    assert(incorrect_label_loss <= loss_incorrect_pred_threshold)

def test_untargeted_cw_loss():
    # Untargeted C&W loss should be positive when a correct prediction is made
    # and negative when an incorrect prediction is made

    # Test a very well-trained model
    pgd_attack = PGDAttack(targeted=False)
    loss_test_helper(
        num_test_classes=10, batch_size=64, loss_correct_pred_threshold=0,
        loss_incorrect_pred_threshold=0, loss_fn=pgd_attack.cw_loss,
        logit_variance=25, logit_mean=0)
    
    # Test a less-confident model (assigns high probabilities to everything)
    pgd_attack = PGDAttack(targeted=False)
    loss_test_helper(
        num_test_classes=10, batch_size=64, loss_correct_pred_threshold=0,
        loss_incorrect_pred_threshold=0, loss_fn=pgd_attack.cw_loss,
        logit_variance=0.01, logit_mean=4)

    # Test another less-confident model (assigns low probabilities to
    # everything)
    pgd_attack = PGDAttack(targeted=False)
    loss_test_helper(
        num_test_classes=10, batch_size=64, loss_correct_pred_threshold=0,
        loss_incorrect_pred_threshold=0, loss_fn=pgd_attack.cw_loss,
        logit_variance=0.01, logit_mean=-4)

def test_cross_entropy_loss():
    # The ce_loss will always be negative, since it is an attack loss (opposite
    # of cross entropy). We want it to be close to zero when labels match logits
    # and very low when labels are different (this is what we want).
    pgd_attack = PGDAttack()
    loss_test_helper(
        num_test_classes=10, batch_size=64, loss_correct_pred_threshold=-0.0001,
        loss_incorrect_pred_threshold=-5, loss_fn=pgd_attack.ce_loss,
        logit_variance=25, logit_mean=0)
