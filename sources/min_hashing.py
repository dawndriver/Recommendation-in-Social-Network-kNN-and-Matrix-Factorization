from const import *
from hash_generator import *
from dict_generator import *
import time

'''
Create the matrix for holding the hash values for each items for signitures
Input F will be from this file this function:
hash_generator.py: gen_minhash_params
'''
def signature_hash_values_for_item(F):
  #create matrix for the hash values for each items
  signature_hash_values = [[0 for x in range(sig_dim)] for x in range(num_items)]
  for i in range (0, num_items):
    for j in range (0, sig_dim):
      a = F[j][0]
      b = F[j][1]
      signature_hash_values[i][j] = (a * i + b) % minhash_prime
  return signature_hash_values

'''
Get the signatures of each user and store into a dict
user_item_dict will be from:
dict_generator.py: get_follow_dict
signature_hash_values will be from: signature_hash_values_for_item
'''
def calculate_signature_for_users(user_item_dict, signature_hash_values):
  user_signature_dict = {}
  for key, value in user_item_dict.iteritems():
    value_list = list(value)
    #print('signature_hash_values')
    #print(signature_hash_values)
    result = []
    result = signature_hash_values[value_list[0]][:]
    for item in range (0, len(value_list)):
      for i in range (0, sig_dim):
        if result[i] > signature_hash_values[value_list[item]][i]:
          result[i] = signature_hash_values[value_list[item]][i]
    user_signature_dict[key] = result
  #print user_signature_dict
  return user_signature_dict

def test_user_signature_dict():
  user_item_dict = {}
  user_item_dict[101] = [0,1,2]
  user_item_dict[102] = [2,4]
  user_item_dict[103] = [1,2,3]
  user_item_dict[104] = [0,3]
  signature_hash_values = [[0 for x in range(3)] for x in range(5)]
  signature_hash_values[0] = [4,2,3];
  signature_hash_values[1] = [2,3,1];
  signature_hash_values[2] = [3,2,1];
  signature_hash_values[3] = [1,4,3];
  signature_hash_values[4] = [2,3,4];
  calculate_signature_for_users(user_item_dict, signature_hash_values)

def min_hash():
  F = gen_minhash_params(sig_dim)
  signature_hash_values_for_item(F)
  follow_dict = get_follow_dict()
  start_time = time.time()
  sigDict = calculate_signature_for_users(follow_dict, signature_hash_values_for_item(F))
  print ("Signature Generation Time: %.2f s" % (time.time()-start_time))
  start_time = time.time()
  write_dict_into_file(sigDict, signature_path)
  print ("Signature File Dump Time: %.2f s" % (time.time()-start_time))

if __name__ == "__main__":
  min_hash()
  '''
  dicts = {}
  dicts['test0'] = [1,2,3]
  dicts['test1'] = [2,3,4]
  dicts['test2'] = [3,4,5]
  write_dict_into_file(dicts, 'result.txt')
  '''
