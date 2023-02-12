"""
Tests to make sure that the sgd update step is correct
"""
from attack_util import PGDAttack
import torch
import random

def loss_test_helper(num_test_classes, batch_size, loss_correct_pred_threshold,
                     loss_incorrect_pred_threshold, loss_fn, logit_variance,
                     logit_mean):
    """
    Helper function to test an untargeted loss.

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

def targeted_lost_test_helper(
    loss_fn, target_class, batch_size, num_test_classes,
    target_class_confidence, default_confidence,
    most_confident_nontarget_confidence, required_upper_bound_on_loss,
    required_lower_bound_on_loss):
    """Helper function to test a targeted loss

    Args:
        loss_fn                             The loss function being tested
        batch_size                          Batch size of dummy input
        num_test_classes                    Number of classes in dummy input
        target_class_confidence             Level of confidence the dummy logit
                                            should have in the target class
        default_confidence                  Baseline level of confidence the
                                            dummy logit should have in all
                                            non-target classes
        most_confident_nontarget_confidence Highest level of confidence the
                                            dummy logit should have in a
        required_upper_bound_on_loss        Tests to make sure the loss is
                                            below this value. Set to None if
                                            the loss has no upper bound.
        required_lower_bound_on_loss        Tests to make sure the loss is above
                                            this value. Set to None if the loss
                                            has no lower bound.
    """
    possible_nontarget_classes = list(range(num_test_classes))
    possible_nontarget_classes.remove(target_class)

    dummy_logits = (
        torch.ones((batch_size, num_test_classes)) * default_confidence)

    for it in range(batch_size):
        dummy_logits[it][target_class] = target_class_confidence
        
        most_confident_nontarget_class = random.choice(
            possible_nontarget_classes)
        
        dummy_logits[it][most_confident_nontarget_class] = (
            most_confident_nontarget_confidence)

    dummy_labels = torch.tensor(
        random.choices(possible_nontarget_classes, k=batch_size))

    if required_upper_bound_on_loss:
        assert((loss_fn(dummy_logits, dummy_labels) <=
            required_upper_bound_on_loss))
    if required_lower_bound_on_loss:
        assert((loss_fn(dummy_logits, dummy_labels) >=
            required_lower_bound_on_loss))


def test_targeted_cw_loss():
    # Can't reuse the helper above, since the target labels take
    # over the role of the true class labels in the untargeted attackk

    pgd_attack = PGDAttack(targeted=True, loss_type='cw')


    # Test model confident in target class
    # Required upper bound on loss is set to 0 so that a model confident in the
    # target class always has a negative loss.
    targeted_lost_test_helper(
        loss_fn=pgd_attack.cw_loss, target_class=1, batch_size=64,
        num_test_classes=10, target_class_confidence=10., default_confidence=5.,
        most_confident_nontarget_confidence=9.5, required_upper_bound_on_loss=0,
        required_lower_bound_on_loss=None)

    # Test model not confident in target class. Loss should be positive, so
    # lower bound is 0.001
    targeted_lost_test_helper(
        loss_fn=pgd_attack.cw_loss, target_class=1, batch_size=64,
        num_test_classes=10, target_class_confidence=4, default_confidence=5.,
        most_confident_nontarget_confidence=7.,
        required_upper_bound_on_loss=None, required_lower_bound_on_loss=0.5)

    # Test another model not confident in target class. Loss should be positive, so
    # lower bound is 0
    targeted_lost_test_helper(
        loss_fn=pgd_attack.cw_loss, target_class=1, batch_size=64,
        num_test_classes=10, target_class_confidence=4, default_confidence=0,
        most_confident_nontarget_confidence=7.,
        required_upper_bound_on_loss=None, required_lower_bound_on_loss=0.5)

def test_untargeted_cw_loss():
    # Untargeted C&W loss should be positive when a correct prediction is made
    # and negative when an incorrect prediction is made

    # Test a very well-trained model
    pgd_attack = PGDAttack(targeted=False, loss_type='cw')
    loss_test_helper(
        num_test_classes=10, batch_size=64, loss_correct_pred_threshold=0,
        loss_incorrect_pred_threshold=0, loss_fn=pgd_attack.cw_loss,
        logit_variance=25, logit_mean=0)
    
    # Test a less-confident model (assigns high probabilities to everything)
    pgd_attack = PGDAttack(targeted=False, loss_type='cw')
    loss_test_helper(
        num_test_classes=10, batch_size=64, loss_correct_pred_threshold=0,
        loss_incorrect_pred_threshold=0, loss_fn=pgd_attack.cw_loss,
        logit_variance=0.01, logit_mean=4)

    # Test another less-confident model (assigns low probabilities to
    # everything)
    pgd_attack = PGDAttack(targeted=False, loss_type='cw')
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
