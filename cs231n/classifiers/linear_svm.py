import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i,:] / num_train
                dW[:,y[i]] -= X[i,:] / num_train

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW = dW + 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    scores = X.dot(W)
    correct_label_scores_idxes = (range(num_train), y) # tuple 모음
    correct_label_scores = scores[correct_label_scores_idxes] # s_y_i만 뽑아냄
    
    score_diff = scores - np.reshape(correct_label_scores,(-1 , 1)) # matrix - vector
    score_diff += 1
    
    score_diff[correct_label_scores_idxes] = 0 # 빼주고나서 y_i번째 자리는 0으로
    
    idx_of_neg = np.nonzero(score_diff < 0) # s_j - s_y_i + 1 < 0 인 곳의 index뽑아내기
    score_diff[idx_of_neg] = 0
    
    loss = score_diff.sum()
    loss /= num_train
    loss += reg * np.sum(W * W)
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    score_diff[score_diff > 0] = 1
    correct_label_grad = score_diff.sum(axis=1) * -1
    score_diff[correct_label_scores_idxes] = correct_label_grad
    
    dW = X.T.dot(score_diff) # dL/dW = X.T x dL/d(score) # score_diff / num_train이 dL/d(score)임
    dW /= num_train
    dW += 2 * reg * W
    
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
