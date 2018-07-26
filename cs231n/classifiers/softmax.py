import numpy as np
from random import shuffle
import math

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    scores -= np.max(scores) # for numeric stability
    
    num_train = X.shape[0]
    num_dim = X.shape[1]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        denom = 0.0
        for j in range(num_classes):
            denom += math.exp(scores[i,j])
        l_i = -1 * scores[i,y[i]] + math.log(denom)
        loss += l_i
        
        for j in range(num_classes):
            dW[:,j] += (1 / denom) * math.exp(scores[i,j]) * X[i]
            if j == y[i]:
                dW[:,j] -= X[i]
    
    loss /= num_train
    dW /= num_train
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * np.sum(W * W)
    
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_dim = X.shape[1]
    num_classes = W.shape[1]
    
    scores = X.dot(W)
    scores -= np.max(scores)
    expscores = np.exp(scores)
    
    denoms = expscores.sum(axis=1)
    numers = expscores[(range(num_train),y)]
    loss = -1 * np.log(numers / denoms).sum()
    
    correct_class = np.zeros(scores.shape)
    correct_class[(range(num_train),y)] = -1
    dW_term1 = X.T.dot(correct_class)
    dW_term2 = (X.T / denoms).dot(expscores)
    dW = dW_term1 + dW_term2
    
    loss /= num_train
    dW /= float(num_train)
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

