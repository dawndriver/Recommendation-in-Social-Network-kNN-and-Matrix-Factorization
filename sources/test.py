from sets import Set
from const import *
from hash_generator import *
from dict_generator import *
from lsh import *
from matrix_factor import *
import numpy as np
import time
import random
import itertools

# ================ Global Vars ==============

item_dict = None        # item id -> item index
actions_dict = None     # user id -> list of followed item indices with actions
keywords_dict = None    # user id -> list of words and weights
user_sig_dict = None    # user id -> user min-hash signature

lsh_buckets_3 = None
lsh_buckets_4 = None
lsh_buckets_5 = None

result_idx = 0

log_sample_size = 10    # the number of sample users to perform the experiment

user_sample_size = 100
max_iter = 500          # max number of iterations to do matrix factorization
regularization = 0.01   # regularization factor in matrix factorization
K = 2                   # the dimension of latent space

use_keywords = True

#hard_limiter_threshold = 0.9   # the threshold to decide whether to recommend or not
#use_relative_limiter = False
#relative_limiter_percentage = 0.25

# ================ Global Vars ==============

def get_log_sample():
  with open(log_out+'4.txt') as fs:
    contents = list(itertools.islice(fs, log_sample_size))
  logs = [[] for i in contents]
  idx = 0
  for line in contents:
     line = line.strip()
     segs = line.split('\t')     
     logs[idx] = (int(segs[0]),int(segs[1]),int(segs[2]))
     idx = idx + 1
  return logs

def redo_lsh(b):
  global user_sig_dict
  global lsh_buckets

  start_time = time.time()
  lsh_buckets = lsh(user_sig_dict, B=b)
  print 'LSH Buckets Creation Time: %.2f s' % (time.time()-start_time)
  
def sample_from_user(n, fromFile=False, log_user_file=sample_log_user):
  logs = {}
  lines = None
  if fromFile:
    with open(log_user_file) as fs:
      lines = fs.readlines()    
  else:
    with open(log_user) as fs:
      contents = fs.readlines()
      lines = random.sample(contents,n)
  for i in lines:
    i = i.strip()
    segs = i.split('\t')
    uid = int(segs[0])
    recommends = segs[1]
    tuples = recommends.split(' ')
    if uid not in logs.keys():
      logs[uid] = []
    for t in tuples:
      itemId = int(t.split(',')[0].strip())
      res = int(t.split(',')[1].strip())
      logs[uid].append((itemId,res))
  return logs

def read_dicts():
  global item_dict 
  global user_sig_dict
  global actions_dict
  global keywords_dict
  global lsh_buckets_3
  global lsh_buckets_4
  global lsh_buckets_5
  
  print ''
  print '=================== Dictionaries Creation ====================='
  print ''
  
  start_time = time.time()
  item_dict = get_item_dict()
  print 'Item Dict Creation Time: %.2fs' % (time.time()-start_time)
  
  start_time = time.time()
  actions_dict = get_actions_dict(item_dict)
  print 'User Actions Dict Creation Time: %.2fs' % (time.time()-start_time)
  
  start_time = time.time()
  keywords_dict = get_keywords_dict()
  print 'User Keywords Dict Creation Time: %.2fs' % (time.time()-start_time)
  
  start_time = time.time()
  user_sig_dict = read_dict_from_file(signature_path)
  print 'User Signature Dict Creation Time: %.2fs' % (time.time()-start_time)
  
  start_time = time.time()
  # lsh_buckets = lsh(user_sig_dict)
  
  # logs = sample_from_user(user_sample_size, True)
  # lsh_buckets_3 = lsh(user_sig_dict, B=3, target_users=logs.keys())
  # lsh_buckets_4 = lsh(user_sig_dict, B=4, target_users=logs.keys())
  # lsh_buckets_5 = lsh(user_sig_dict, B=5, target_users=logs.keys())
  
  lsh_buckets_3 = lsh(user_sig_dict, B=3)
  lsh_buckets_4 = lsh(user_sig_dict, B=4)
  lsh_buckets_5 = lsh(user_sig_dict, B=5)
  print 'LSH Buckets Creation Time: %.2fs' % (time.time()-start_time)

def RandExperiment():  # experiment by random guess
  # prepare log samples
  logs = sample_from_user(user_sample_size, True)
  
  # accumulators for tally report
  true_positives = 0
  true_negatives = 0
  false_positives = 0
  false_negatives = 0
  
  for userId in logs.keys():
    # if userId not in user_sig_dict.keys():
      # print '    He does not follow anyone previously, ignore\n'
      # continue
    
    for (itemId,result) in logs[userId]:
      decision = random.sample([0,1,1,1],1)[0]
      if result == 1:
        if decision == 1:
          true_positives = true_positives + 1
        else: # we miss a good recommend, Type II error
          false_negatives = false_negatives + 1
      elif result == -1:
        if decision == 0:
          true_negatives = true_negatives + 1
        else: # we treat a bad recommend as good, Type I error
          false_positives = false_positives + 1
     
  # print out results
  sensitivity = true_positives * 1.0 / (true_positives + false_negatives) # hit rate (1-miss rate) or recall
  precision = true_positives * 1.0 / (true_positives + false_positives) # positive predictive value (PPV)
  F_score = 2 * (precision * sensitivity) / (precision + sensitivity)
  
  resStr = 'True Positives:%d, True Negatives:%d, False Positives:%d, False Negatives:%d\n' % (true_positives,true_negatives,false_positives,false_negatives)
  resStr = resStr + 'Precision(PPV):%.2f, Sensitivity(Recall):%.2f, F_Score:%.2f' % (precision,sensitivity,F_score)
  print resStr
  
def PureKNNExperiment():   # experiment by using Collaborative Filtering for KNN
  # prepare log samples
  logs = sample_from_user(user_sample_size, True)
  
  hard_limits = [1,2,3,4,5]
  
  # accumulators for tally report
  limits_count = len(hard_limits)
  true_positives = [0 for i in range(limits_count) ]
  true_negatives = [0 for i in range(limits_count) ]
  false_positives = [0 for i in range(limits_count) ]
  false_negatives = [0 for i in range(limits_count) ]
  
  for userId in logs.keys():
    print 'For user %d:' % userId
    print '----------------------------------------'
    
    if userId not in user_sig_dict.keys():
      print '    He does not follow anyone previously, ignore\n'
      continue

    start_time = time.time()
    neighbours = lsh_query_optimize(lsh_buckets_3,lsh_buckets_4,lsh_buckets_5,user_sig_dict[userId])
    # neighbours = lsh_query(lsh_buckets, user_sig_dict[userId])
    print '    Found neighbours count: %d with time %fs' % (len(neighbours),time.time()-start_time)
    
    if userId not in neighbours:  # add target user in case we miss it in random sampling
      neighbours.add(userId)
    
    if len(neighbours) > max_neighbours_sim:
      print '        Neighbours so many, reduce to %d' % max_neighbours_sim
      neighbours = random.sample(neighbours, max_neighbours_sim-1)
      if userId not in neighbours:
        neighbours.append(userId)
    elif len(neighbours) < K:   # the user may either did not follow anyone or so unique in taste, just ignore for now
      print '        Too few neighbours, ignore'
      continue
    
    userIdxDict,userIdxDictRev,R = getUtilityMatrix(item_dict,actions_dict,neighbours)
    
    start_idx = 0;
    for limit in hard_limits:
      for (itemId,result) in logs[userId]:
        recommend = np.sum(R[:,item_dict[itemId]]) >= limit     # meaning there is a similar user likes it, so recommend this
        if result == 1:
          if recommend:
            true_positives[start_idx] = true_positives[start_idx] + 1
          else: # we miss a good recommend, Type II error
            false_negatives[start_idx] = false_negatives[start_idx] + 1
        elif result == -1:
          if not recommend:
            true_negatives[start_idx] = true_negatives[start_idx] + 1
          else: # we treat a bad recommend as good, Type I error
            false_positives[start_idx] = false_positives[start_idx] + 1
      start_idx = start_idx + 1
  
  start_idx = 0;
  printStatus()
  for limit in hard_limits:
    # print out results
    sensitivity = true_positives[start_idx] * 1.0 / (true_positives[start_idx] + false_negatives[start_idx]) # hit rate (1-miss rate) or recall
    precision = true_positives[start_idx] * 1.0 / (true_positives[start_idx] + false_positives[start_idx]) # positive predictive value (PPV)
    resStr = 'Limit: %d\n' % limit
    resStr = resStr + 'True Positives:%d, True Negatives:%d, False Positives:%d, False Negatives:%d\n' % (true_positives[start_idx],true_negatives[start_idx],false_positives[start_idx],false_negatives[start_idx])
    resStr = resStr + 'Precision(PPV):%.2f, Sensitivity(Recall):%.2f\n\n' % (precision,sensitivity)
    print resStr
    start_idx = start_idx + 1
  
def PureMFExperiment():
  item_dict = get_item_dict()
  actions_dict = get_actions_dict(item_dict)
  
  # prepare log samples
  logs = sample_from_user(user_sample_size, True)
  logs2 = sample_from_user(user_sample_size, True, 'data/sample_log_user_1000.txt')
  
  print ''
  print '=================== Experiment Start ======================'
  print ''
  
  userIdxDict,userIdxDictRev,R = getUtilityMatrix(item_dict,actions_dict,set(logs.keys()).union(set(logs2.keys())))
  
  N = len(R)
  M = len(R[0])
  initialP = np.random.rand(N,K)
  initialQ = np.random.rand(M,K)
    
  print '    ==== Start Matrix Factorization ===='
  
  start_time = time.time()
  start_err,end_err,P,Q = matrix_factorization(R,initialP,initialQ,K,alpha=0.1,steps=max_iter,lamda=regularization)
  print 'Matrix Factorization finish with %.3fs and error %.2f -> %.2f' % (time.time()-start_time,start_err,end_err)
  
  hard_limits = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4, 4.5, 5.0]
  
  # accumulators for tally report
  limits_count = len(hard_limits)
  true_positives = [0 for i in range(limits_count) ]
  true_negatives = [0 for i in range(limits_count) ]
  false_positives = [0 for i in range(limits_count) ]
  false_negatives = [0 for i in range(limits_count) ]
  
  for userId in logs.keys():
    user_latent_vec = P[userIdxDict[userId],:]
    start_idx = 0
    for hard_limiter_threshold in hard_limits:
      for (itemId,result) in logs[userId]:
        item_latent_vec = Q[item_dict[itemId],:]
        cal_result = np.dot(user_latent_vec,item_latent_vec)
        if result == 1:
          if cal_result >= hard_limiter_threshold:
            true_positives[start_idx] = true_positives[start_idx] + 1
          else: # we miss a good recommend, Type II error
            false_negatives[start_idx] = false_negatives[start_idx] + 1
        elif result == -1:
          if cal_result < hard_limiter_threshold:
            true_negatives[start_idx] = true_negatives[start_idx] + 1
          else: # we treat a bad recommend as good, Type I error
            false_positives[start_idx] = false_positives[start_idx] + 1
      start_idx = start_idx + 1

  # print out results
  resStr = getStatusStr() + '\n'
  start_idx = 0
  for val in hard_limits:
    resStr = resStr + 'HardLimit: %.2f\n--------------------\n' % val
    resStr = resStr + 'True Positives:%d, True Negatives:%d, False Positives:%d, False Negatives:%d\n' % (true_positives[start_idx],true_negatives[start_idx],false_positives[start_idx],false_negatives[start_idx])
    sensitivity = true_positives[start_idx] * 1.0 / (true_positives[start_idx] + false_negatives[start_idx]) # hit rate (1-miss rate) or recall
    precision = true_positives[start_idx] * 1.0 / (true_positives[start_idx] + false_positives[start_idx]) # positive predictive value (PPV)
    resStr = resStr + 'Precision(PPV):%.3f, Sensitivity(Recall):%.3f\n\n' % (precision,sensitivity)
    start_idx = start_idx + 1
  print resStr
  with open('temp/result'+str(result_idx)+'.txt','w+') as fs:
    fs.write(resStr)

def MFExperiment():   # experiment by using Matrix Factorization for KNN
  # prepare log samples
  logs = sample_from_user(user_sample_size, True)
  
  print ''
  print '=================== Experiment Start ======================'
  print ''
  
  limiter_selector = [False]
  hard_limits = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
  relative_limits = [0.50, 0.40, 0.30, 0.20, 0.10]
  
  # accumulators for tally report
  efficiencies = []
  limits_count = len(relative_limits) + len(hard_limits)
  true_positives = [0 for i in range(limits_count) ]
  true_negatives = [0 for i in range(limits_count) ]
  false_positives = [0 for i in range(limits_count) ]
  false_negatives = [0 for i in range(limits_count) ]
  
  for userId in logs.keys():
    print 'For user %d:' % userId
    print '----------------------------------------'
    
    if userId not in user_sig_dict.keys():
      print '    He does not follow anyone previously, ignore\n'
      continue

    start_time = time.time()
    neighbours = lsh_query_optimize(lsh_buckets_3,lsh_buckets_4,lsh_buckets_5,user_sig_dict[userId])
    # neighbours = lsh_query(lsh_buckets, user_sig_dict[userId])
    print '    Found neighbours count: %d with time %fs' % (len(neighbours),time.time()-start_time)
    
    if userId not in neighbours:  # add target user in case we miss it in random sampling
      neighbours.add(userId)
    
    if len(neighbours) > max_neighbours_sim:
      print '        Neighbours so many, reduce to %d' % max_neighbours_sim
      neighbours = random.sample(neighbours, max_neighbours_sim-1)
      if userId not in neighbours:
        neighbours.append(userId)
    elif len(neighbours) < K:   # the user may either did not follow anyone or so unique in taste, just ignore for now
      print '        Too few neighbours, ignore'
      continue
    
    start_time = time.time()
    userIdxDict,userIdxDictRev,R = getUtilityMatrix(item_dict,actions_dict,neighbours)
    
    N = len(R)
    M = len(R[0])
    initialP = np.random.rand(N,K)
    initialQ = np.random.rand(M,K)
    
    print '\n    ==== Start Matrix Factorization ===='
    
    if not use_keywords: # classical matrix factorization
      start_err,end_err,P,Q = matrix_factorization(R,initialP,initialQ,K,steps=max_iter,lamda=regularization)
      training_time = time.time() - start_time
      efficiencies.append(training_time / len(neighbours))
      print '    ==== End Matrix Factorization ====\n'
      print '    Matrix Factorization takes %.2fs with error %.2f -> %.2f\n' % (training_time,start_err,end_err)
      
      user_latent_vec = P[userIdxDict[userId],:]

      for use_relative_limiter in limiter_selector:
        if use_relative_limiter:   # use relative limiter
          start_idx = 0
          for relative_limiter_percentage in relative_limits:
            user_item_vec = np.dot(user_latent_vec, Q.T)
            topN = int(len(user_item_vec)*relative_limiter_percentage)
            topItemIndices = np.argsort(user_item_vec)[-topN:]
            for (itemId,result) in logs[userId]:
              if result == 1:
                if item_dict[itemId] in topItemIndices:
                  true_positives[start_idx] = true_positives[start_idx] + 1
                else: # we miss a good recommend, Type II error
                  false_negatives[start_idx] = false_negatives[start_idx] + 1
              elif result == -1:
                if item_dict[itemId] not in topItemIndices:
                  true_negatives[start_idx] = true_negatives[start_idx] + 1
                else: # we treat a bad recommend as good, Type I error
                  false_positives[start_idx] = false_positives[start_idx] + 1
            start_idx = start_idx + 1
        else:   # use hard limiter
          start_idx = len(relative_limits)
          for hard_limiter_threshold in hard_limits:
            for (itemId,result) in logs[userId]:
              item_latent_vec = Q[item_dict[itemId],:]
              cal_result = np.dot(user_latent_vec,item_latent_vec)
              if result == 1:
                if cal_result >= hard_limiter_threshold:
                  true_positives[start_idx] = true_positives[start_idx] + 1
                else: # we miss a good recommend, Type II error
                  false_negatives[start_idx] = false_negatives[start_idx] + 1
              elif result == -1:
                if cal_result < hard_limiter_threshold:
                  true_negatives[start_idx] = true_negatives[start_idx] + 1
                else: # we treat a bad recommend as good, Type I error
                  false_positives[start_idx] = false_positives[start_idx] + 1
            start_idx = start_idx + 1
            
    else:   # optimized matrix factorization with user keywords inside
      keywords_dict_for_matrix = {}    # matrix row i -> list of (keyword latent vector, keyword weight)
      for uid in neighbours:
        keywords_dict_for_matrix[userIdxDict[uid]] = []
      for uid in neighbours:
        for (wid,wt) in keywords_dict[uid]:
          keywords_dict_for_matrix[userIdxDict[uid]].append((np.random.rand(K),wt))
      
      start_err,end_err,P,Q = matrix_factorization2(R,initialP,initialQ,K,keywords_dict_for_matrix,steps=max_iter,lamda=regularization)
      training_time = time.time() - start_time
      efficiencies.append(training_time / len(neighbours))
      print '    ==== End Matrix Factorization ===='
      print '    Finish Matrix Factorization with time %.2fs and error %.2f -> %.2f\n' % (training_time,start_err,end_err)
      
      user_latent_vec = P[userIdxDict[userId],:]
      for wd,wt in keywords_dict_for_matrix[userIdxDict[userId]]:
        user_latent_vec = user_latent_vec + wd * wt 
      
      for use_relative_limiter in limiter_selector:
        if use_relative_limiter:   # use relative limiter
          start_idx = 0
          for relative_limiter_percentage in relative_limits:
            user_item_vec = np.dot(user_latent_vec, Q.T)
            topN = int(len(user_item_vec)*relative_limiter_percentage)
            topItemIndices = np.argsort(user_item_vec)[-topN:]
            for (itemId,result) in logs[userId]:
              if result == 1:
                if item_dict[itemId] in topItemIndices:
                  true_positives[start_idx] = true_positives[start_idx] + 1
                else: # we miss a good recommend, Type II error
                  false_negatives[start_idx] = false_negatives[start_idx] + 1
              elif result == -1:
                if item_dict[itemId] not in topItemIndices:
                  true_negatives[start_idx] = true_negatives[start_idx] + 1
                else: # we treat a bad recommend as good, Type I error
                  false_positives[start_idx] = false_positives[start_idx] + 1
            start_idx = start_idx + 1
        else:   # use hard limiter
          start_idx = len(relative_limits)
          for hard_limiter_threshold in hard_limits:
            for (itemId,result) in logs[userId]:
              item_latent_vec = Q[item_dict[itemId],:]
              cal_result = np.dot(user_latent_vec,item_latent_vec)
              if result == 1:
                if cal_result >= hard_limiter_threshold:
                  true_positives[start_idx] = true_positives[start_idx] + 1
                else: # we miss a good recommend, Type II error
                  false_negatives[start_idx] = false_negatives[start_idx] + 1
              elif result == -1:
                if cal_result < hard_limiter_threshold:
                  true_negatives[start_idx] = true_negatives[start_idx] + 1
                else: # we treat a bad recommend as good, Type I error
                  false_positives[start_idx] = false_positives[start_idx] + 1
            start_idx = start_idx + 1
  
  # print out results
  resStr = getStatusStr() + '\n'
  resStr = resStr + 'Efficiency Avg:%.3f Std:%.3f\n-----------------\n' % (np.average(efficiencies),np.std(efficiencies))
  for use_relative_limiter in limiter_selector:
    if use_relative_limiter:
      start_idx = 0
      for val in relative_limits:
        resStr = resStr + 'RelativeLimit: %.2f\n--------------------\n' % val
        resStr = resStr + 'True Positives:%d, True Negatives:%d, False Positives:%d, False Negatives:%d\n' % (true_positives[start_idx],true_negatives[start_idx],false_positives[start_idx],false_negatives[start_idx])
        sensitivity = true_positives[start_idx] * 1.0 / (true_positives[start_idx] + false_negatives[start_idx]) # hit rate (1-miss rate) or recall
        precision = true_positives[start_idx] * 1.0 / (true_positives[start_idx] + false_positives[start_idx]) # positive predictive value (PPV)
        F_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        resStr = resStr + 'Precision(PPV):%.3f, Sensitivity(Recall):%.3f, F_Score:%.3f\n\n' % (precision,sensitivity,F_score)
        start_idx = start_idx + 1
    else:
      start_idx = len(relative_limits)
      for val in hard_limits:
        resStr = resStr + 'HardLimit: %.2f\n--------------------\n' % val
        resStr = resStr + 'True Positives:%d, True Negatives:%d, False Positives:%d, False Negatives:%d\n' % (true_positives[start_idx],true_negatives[start_idx],false_positives[start_idx],false_negatives[start_idx])
        sensitivity = true_positives[start_idx] * 1.0 / (true_positives[start_idx] + false_negatives[start_idx]) # hit rate (1-miss rate) or recall
        precision = true_positives[start_idx] * 1.0 / (true_positives[start_idx] + false_positives[start_idx]) # positive predictive value (PPV)
        F_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        resStr = resStr + 'Precision(PPV):%.3f, Sensitivity(Recall):%.3f, F_Score:%.3f\n\n' % (precision,sensitivity,F_score)
        start_idx = start_idx + 1

  print resStr
  with open('temp/result'+str(result_idx)+'.txt','w+') as fs:
    fs.write(resStr)

def getStatusStr():
  resStr = '*******************************************************************************' + '\n'
  resStr = resStr + '*** Iterations:%d, UseKeyWords:%s\n' % (max_iter,use_keywords)
  resStr = resStr + '*** MaxNeighbours:%d, Regularization:%.2f, LatentDimension:%d, SampleUsers:%d\n' % (max_neighbours_sim,regularization,K,user_sample_size)
  resStr = resStr + '*******************************************************************************'
  return resStr
  
def printStatus():
  print getStatusStr()
  
if __name__ == '__main__':
  result_idx = 0
  user_sample_size = 100
  use_keywords = False
  max_iter = 2000
  
  result_idx = 10
  max_neighbours_sim = 200
  #printStatus()
  #read_dicts()
  #MFExperiment()
  #PureKNNExperiment()
  PureMFExperiment()
  #RandExperiment()