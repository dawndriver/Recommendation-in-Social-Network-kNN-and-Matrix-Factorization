from const import *
import numpy as np
import math
import warnings
warnings.filterwarnings('error')
from dict_generator import *

def getUtilityMatrix(itemDict,actions_dict,userSet):
  userIdxDict = {}      # user_id -> row_idx of matrix
  userIdxDictRev = {}   # row_idx -> user_id of matrix
  
  idx = 0
  for uid in userSet:
    userIdxDict[uid] = idx
    userIdxDictRev[idx] = uid
    idx += 1
  
  actionKeySet = set(actions_dict.keys())
  
  M = np.zeros(shape=(len(userSet),len(itemDict.keys())))
  for userId in userSet:
    if userId in actionKeySet:  # the user needs to have action to show the interest
      for (itemIdx,atActions,retweetActions,commentActions) in actions_dict[userId]:
        activities = atActions + commentActions + retweetActions
        M[userIdxDict[userId]][itemIdx] = activities
  
  # initialize utility matrix
  # with open(action_path) as fs:
  	 # contents = fs.readlines()
  # for line in contents:
    # line = line.strip()
    # segs = line.split('\t')
    # srcUser = int(segs[0])
    # destUser = int(segs[1])
    # if srcUser in userSet and destUser in itemDict.keys():
      # atActions = int(segs[2])
      # retweetActions = int(segs[3])
      # commentActions = int(segs[4])
      # activities = atActions * 10 + commentActions * 5 + retweetActions  # put different weights for different actions
      # M[userIdxDict[srcUser]][itemDict[destUser]] = activities
  
  return userIdxDict,userIdxDictRev,M

"""
@INPUT:
    R     : utility matrix to be factorized, of dimension N x M
    		R[i][j] means the rating of user i to item j
    P     : user latent matrix of dimension N x K
    		P[i][j] means the relation strength of user i to latent factor j
    Q     : item latent matrix of dimension M x K
    		Q[i][j] means the relation strength of item i to latent factor j
    K     : the number of latent features
    steps : the maximum number of steps in training
    alpha : the learning rate
    lamda  : the regularization parameter
@OUTPUT:
    the trained matrices P and Q
"""
def matrix_factorization(R,P,Q,K,steps=5000,alpha=0.05,lamda=0.01,check_step=10,dec_mul=0.5,inc_add=0.00001):
  fs = open('temp/error_decrease.txt', 'w+')
  Q = Q.T
  start_err = eval_matrix(R,P,Q,K,lamda)
  pre_e = start_err
  pre_P = np.copy(P)
  pre_Q = np.copy(Q)
  check_next = False
  for step in xrange(steps):
    for i in xrange(len(R)):
      for j in xrange(len(R[i])):
        if R[i][j] > 0:
          eij = R[i][j] - np.dot(P[i,:],Q[:,j])
          
          if math.isnan(eij):
            P = np.copy(pre_P)
            Q = np.copy(pre_Q)
            alpha = alpha * dec_mul
            check_next = True
            eij = R[i][j] - np.dot(P[i,:],Q[:,j])
            print '    !!!! Found learning rate too high! Decrease learning rate to be %f' % alpha
          
          try:
            for k in xrange(K):
              P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - lamda * P[i][k])
              Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - lamda * Q[k][j])
          except RuntimeWarning: # double overflow, meaning learnig rate too larege
            P = np.copy(pre_P)
            Q = np.copy(pre_Q)
            alpha = alpha * dec_mul
            check_next = True
            print '    !!!! Found learning rate too high! Decrease learning rate to be %f' % alpha
    
    if check_next or step % check_step == 0: # error check point
      try:
        e = eval_matrix(R,P,Q,K,lamda)
        if math.isnan(e) or e >= pre_e: # the learning rate chosen to be too large!
          P = np.copy(pre_P)
          Q = np.copy(pre_Q)
          alpha = alpha * dec_mul
          check_next = True
          print '    !!!! Found learning rate too high! Decrease learning rate to be %f' % alpha
        else: # current learning rate is okay, slowly increase
          fs.write("LR:%f Err:%f\n" % (alpha,e))
          alpha = alpha + inc_add
          pre_e = e
          pre_P = np.copy(P)
          pre_Q = np.copy(Q)
          check_next = False
      except RuntimeWarning:
        P = np.copy(pre_P)
        Q = np.copy(pre_Q)
        alpha = alpha * dec_mul
        check_next = True
        print '    !!!! Found learning rate too high! Decrease learning rate to be %f' % alpha
  
  end_err = eval_matrix(R,P,Q,K,lamda)
  fs.close()
  return start_err,end_err,P,Q.T
  
def eval_matrix(R,P,Q,K,lamda):
  e = 0
  for i in xrange(len(R)):
    for j in xrange(len(R[i])):
      if R[i][j] > 0:
        e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
        for k in xrange(K):
          e = e + (lamda/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
  return e
  
def copy_keywords_dict(keywords_dict):
  new_dict = {}
  for i in keywords_dict.keys():
    new_dict[i] = []
    words = keywords_dict[i]
    for wd,wt in words:
      new_dict[i].append((np.copy(wd),wt))
  return new_dict
  
def matrix_factorization2(R,P,Q,K,keywords_dict,steps=500,alpha=0.05,lamda=0.01,check_step=20,dec_mul=0.5,inc_add=0.00001):
  Q = Q.T
  start_err = eval_matrix2(R,P,Q,K,keywords_dict,lamda)
  pre_e = start_err
  pre_P = np.copy(P)
  pre_Q = np.copy(Q)
  pre_Keywords = copy_keywords_dict(keywords_dict)
  check_next = False
  for step in xrange(steps):
    for i in xrange(len(R)):
      for j in xrange(len(R[i])):
        if R[i][j] > 0:
          words = keywords_dict[i]
          newP = P[i,:]
          for wd,wt in words:
            newP = newP + wd * wt
          eij = R[i][j] - np.dot(newP,Q[:,j])
          if math.isnan(eij):
            P = np.copy(pre_P)
            Q = np.copy(pre_Q)
            keywords_dict = copy_keywords_dict(pre_Keywords)
            alpha = alpha * dec_mul
            check_next = True
            print '    !!!! Found learning rate too high! Decrease learning rate to be %f' % alpha
            words = keywords_dict[i]
            newP = P[i,:]
            for wd,wt in words:
              newP = newP + wd * wt
            eij = R[i][j] - np.dot(newP,Q[:,j])
          
          try:
            for k in xrange(K):
              sum = 0
              for wd,wt in words:
                sum = sum + wd[k] * wt
            
              P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - lamda * P[i][k])
              Q[k][j] = Q[k][j] + alpha * (2 * eij * (P[i][k]+sum) - lamda * Q[k][j])
              for wd,wt in words:
                wd[k] = wd[k] + alpha * (2 * eij * Q[k][j] - lamda * wd[k])
          except RuntimeWarning:
            P = np.copy(pre_P)
            Q = np.copy(pre_Q)
            keywords_dict = copy_keywords_dict(pre_Keywords)
            alpha = alpha * dec_mul
            check_next = True
            print '    !!!! Found learning rate too high! Decrease learning rate to be %f' % alpha
    
    if check_next or step % check_step == 0: # error check point
      try:
        e = eval_matrix2(R,P,Q,K,keywords_dict,lamda)
        if math.isnan(e) or e >= pre_e: # the learning rate chosen to be too large!
          P = np.copy(pre_P)
          Q = np.copy(pre_Q)
          keywords_dict = copy_keywords_dict(pre_Keywords)
          alpha = alpha * dec_mul
          check_next = True
          print '    !!!! Found learning rate too high! Decrease learning rate to be %f' % alpha
        else: # current learning rate is okay, slowly increase
          alpha = alpha + inc_add
          pre_e = e
          pre_P = np.copy(P)
          pre_Q = np.copy(Q)
          pre_Keywords = copy_keywords_dict(keywords_dict)
          check_next = False
      except RuntimeWarning:
        P = np.copy(pre_P)
        Q = np.copy(pre_Q)
        keywords_dict = copy_keywords_dict(pre_Keywords)
        alpha = alpha * dec_mul
        check_next = True
        print '    !!!! Found learning rate too high! Decrease learning rate to be %f' % alpha
  
  end_err = eval_matrix2(R,P,Q,K,keywords_dict,lamda)
  return start_err,end_err,P,Q.T

def eval_matrix2(R,P,Q,K,keywords_dict,lamda):
  e = 0
  for i in xrange(len(R)):
    for j in xrange(len(R[i])):
      if R[i][j] > 0:
        words = keywords_dict[i]
        newP = P[i,:]
        for wd,wt in words:
          newP = newP + wd * wt

        e = e + pow(R[i][j] - np.dot(newP,Q[:,j]), 2)
        
        for k in xrange(K):
          sum = 0
          for wd,wt in words:
            sum = sum + pow(wd[k],2) * wt
        
          e = e + (lamda/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) + sum)
  return e

if __name__ == '__main__':
  R = [
        [5,3,0,1],
        [4,0,0,1],
        [1,1,0,5],
        [1,0,0,4],
        [0,1,5,4],
       ]

  R = np.array(R)

  N = len(R)
  M = len(R[0])
  K = 2

  P = np.random.rand(N,K)
  Q = np.random.rand(M,K)

  e, nP, nQ = matrix_factorization(R, P, Q, K, steps=1000, alpha=0.001, lamda=0.01)

  print np.dot(nP,nQ.T)
