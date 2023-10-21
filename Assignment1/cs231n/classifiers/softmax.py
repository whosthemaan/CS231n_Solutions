from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_training = X.shape[0]

    # for each sample input calculate loss and update loss and gradient
    for i in range(num_training):
      scores = np.dot(X[i], W)

      # softmax is different from svm w.r.t its exponential component - numerically stable exponent
      total = np.exp(scores-np.max(scores))
      
      sm = total/np.sum(total)

      #update loss with normalized log function 
      loss -= np.log(sm[y[i]])  
      sm[y[i]] -= 1
      dW += np.multiply.outer(X[i].ravel(), sm.ravel())

    # update loss and gradient with regularization term
    loss = loss/num_training + (np.sum(W*W)*reg)
    dW = dW/num_training + (reg*2*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)****

    num_training = X.shape[0]

    # for each sample input calculate loss and update loss and gradient
    scores = np.dot(X, W)

    # softmax is different from svm w.r.t its exponential component - numerically stable exponent
    stable = np.exp(scores-np.max(scores))
    sm = stable
    sm /= np.sum(stable, 1).reshape(-1,1)
    loss = np.sum(-np.log(stable[range(num_training), y]))

    # update loss and gradient with regularization term
    loss = loss/num_training + (np.sum(W*W)*reg)

    # update the gradient and add regularization term
    sm[range(num_training), y] -= 1
    dW = np.dot(X.T, sm)
    dW = dW/num_training + (reg*2*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
