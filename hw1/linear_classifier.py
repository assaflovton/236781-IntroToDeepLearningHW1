import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.
        self.weights = torch.empty((n_features, n_classes)).normal_(mean=0.0, std=weight_std)
        # W = (𝐷+1)×𝐶

        # ====== YOUR CODE: ======

        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        # ====== YOUR CODE: ======
        # calculate the classes scores matrix
        class_scores = torch.matmul(x, self.weights)
        # choose the best class for each row (sample)
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        num_of_elements = torch.numel(y)
        num_of_non_zero = num_of_elements - torch.nonzero(y - y_pred, as_tuple=False).size(0)
        acc = num_of_non_zero / num_of_elements
        # ========================

        return acc * 100

    def train(
            self,
            dl_train: DataLoader,
            dl_valid: DataLoader,
            loss_fn: ClassifierLoss,
            learn_rate=0.1,
            weight_decay=0.001,
            max_epochs=100,
    ):
        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        # TODO:
        #  Implement model training loop.
        #  1. At each epoch, evaluate the model on the entire training set
        #     (batch by batch) and update the weights.
        #  2. Each epoch, also evaluate on the validation set.
        #  3. Accumulate average loss and total accuracy for both sets.
        #     The train/valid_res variables should hold the average loss
        #     and accuracy per epoch.
        #  4. Don't forget to add a regularization term to the loss,
        #     using the weight_decay parameter.

        # ====== YOUR CODE: ======
        for i in range(max_epochs):  # Epoch: traverse all samples
            acc, l = 0, 0
            acc2, l2 = 0, 0
            for sample, label in dl_train:
                y_hat, x_scores = self.predict(sample)
                l += loss_fn.loss(sample, label, x_scores, y_hat)
                grad = loss_fn.grad()  # first term
                grad += self.weights * weight_decay  # second term
                self.weights -= grad * learn_rate
                acc += self.evaluate_accuracy(label, y_hat)
            train_res.accuracy.append(acc / len(dl_train)), train_res.loss.append(l / len(dl_train))
            for sample, label in dl_valid:
                y_hat, x_scores = self.predict(sample)
                l2 += loss_fn.loss(sample, label, x_scores, y_hat) +torch.norm(self.weights)*(weight_decay/2)
                acc2 += self.evaluate_accuracy(label, y_hat)
            valid_res.accuracy.append(acc2 / len(dl_valid)), valid_res.loss.append(l2 / len(dl_valid))
            # ========================
        print(".", end="")

        print("")
        return train_res, valid_res


    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        add_one = int(has_bias)
        w_images = self.weights[add_one:].T.reshape((self.n_classes,) + img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.001, learn_rate=0.01, weight_decay=0.001)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ===

    # ========================

    return hp
