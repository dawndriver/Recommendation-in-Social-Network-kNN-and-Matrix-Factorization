from sets import Set
from const import *
from hash_generator import *
from dict_generator import *
from lsh import *
from matrix_factor import *
import numpy as np
import time

def similarity_sets(set_a, set_b):

  return len(set_a.intersection(set_b)) / float(len(set_a.union(set_b)));

def find_similarities(uid, set_neighbours, follow_dict):
  result = {}
  set_a = Set(follow_dict[uid])
  for neighbour in set_neighbours:
    set_b = Set(follow_dict[neighbour])
    result[neighbour] = similarity_sets(set_a, set_b)
  return result

if __name__ == '__main__':

  follow_dict = read_dict_from_file(follow_path)
  item_dict = get_item_dict()    # item id -> item idx (column idx of utility matrix)
  
  #user_sig_dict = read_dict_from_file(signature_path)
  query_uid = 2135320
  #lsh_result = []
  #lsh_result = lsh_query(user_sig_dict,query_uid)
  #print "LSH returns those similar items with below id: "
  #print lsh_result

  neighbours = lsh_main(query_uid)
  #neighbours = ([1598468, 667142, 427528, 2215945, 331786, 881197, 1718802, 1843988, 384814, 1112854, 1951000, 1840900, 2163226, 1073948, 1313585, 1010465, 1123163, 1464582, 840743, 1720620, 1264685, 2173486, 2248669, 1757232, 1558024, 223454, 1271862, 1605943, 1635386, 1043259, 2413884, 1196426, 1366335, 1970753, 1390402, 2310723, 1424452, 820191, 1557063, 1483594, 122704, 1778773, 239958, 1833051, 232544, 777314, 971875, 1748581, 1044625, 954990, 1871983, 1008756, 1336440,1050490, 704635, 2279044, 2178438, 1226121, 1531530, 2281101, 748430, 2154640, 767633, 376942, 2148759, 1512081, 112794, 994971, 1124860, 642442, 100001, 672420, 898984, 1705244, 1506221, 1275056, 869044, 2397044, 620219, 2110652, 1378250, 1081534, 1547713, 1323458, 517040, 924868, 1718213, 1379530, 1836236, 1294541, 987095, 1686843, 920284, 167645, 1955550, 1759455, 1696225, 2334185, 2171373, 1603855, 907507, 577269, 1826294, 2200489, 2020604])
  print("neighbours are:")
  print(neighbours)
  print("Starting to calculate the real similarities...")
  #print(find_similarities(query_uid, neighbours, follow_dict))
  '''
  result = find_similarities(query_uid, neighbours, follow_dict)
  max_result = 0
  key_result = -1
  for key, value in result.iteritems():
    if max_result < value and key != query_uid:
      max_result = value
      key_result = key

  print(key_result)
  print(max_result)
  '''

  userIdxDict,userIdxDictRev,R  = getUtilityMatrix(item_dict,follow_dict,neighbours)
  N = len(R)
  M = len(R[0])
  K = 2

  initialP = np.random.rand(N,K)
  initialQ = np.random.rand(M,K)

  start_time = time.time()
  e1,e2,P,Q = matrix_factorization(R,initialP,initialQ,K,steps=100,alpha=0.001,lamda=0.01)
  print "Matrix Factorization Time: %.2f with error: (%.3f, %.3f)" % (time.time()-start_time,e1,e2)

  #print P
  #print Q

