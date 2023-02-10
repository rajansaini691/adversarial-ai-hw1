"""
Tests to make sure that the sgd update step is correct
"""
from attack_util import PGDAttack
import torch

def test_cross_entropy_loss():
    # Defines test dimensions
    num_test_classes = 10
    batch_size = 64

    # Ensures dummy logits are slightly above 0
    label_smoothing_value = 0.0001      

    # For correct predictions, loss should be above this val
    loss_correct_pred_threshold = -0.0001

    # For incorrect predictions, loss should be below this val
    loss_incorrect_pred_threshold = -5

    pgd_attack = PGDAttack()

    dummy_labels_correct = torch.arange(0, batch_size) % num_test_classes
    dummy_labels_incorrect = torch.cat(
        (torch.arange(1, batch_size), torch.tensor([0]))) % num_test_classes

    # Logits should match the correct labels. The corresponding
    # logit should be high for the correct class and low for all
    # others (e.g. if the class is 3, then logits could be
    # [-25, -25, -25, 25, -25, ...])
    dummy_logits = torch.nn.functional.one_hot(
        dummy_labels_correct, num_classes=num_test_classes)
    dummy_logits = (dummy_logits - 1/2) * 50

    correct_label_loss = pgd_attack.ce_loss(dummy_logits, dummy_labels_correct)
    incorrect_label_loss = pgd_attack.ce_loss(dummy_logits, dummy_labels_incorrect)

    # The ce_loss will always be negative, since it is an attack loss (opposite
    # of cross entropy). We want it to be close to zero when labels match logits
    # and very low when labels are different (this is what we want).
    assert(correct_label_loss >= loss_correct_pred_threshold)
    assert(incorrect_label_loss <= loss_incorrect_pred_threshold)
