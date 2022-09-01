from const import *
from min_hashing import *
from matrix_factor import *
import time
import numpy

def lsh_hash(sig):
  h = 0
  for i in sig:
    h = 31*h + i
  return h

def lsh(user_sigs, B = band_width, target_users = None):
  num_bands = sig_dim / B
  buckets_array = [{} for i in range(num_bands)]
  
  if target_users == None:
    for uid,sig in user_sigs.iteritems():
      for i in range(num_bands):
        si = i*B
        band_sig = sig[si:si+B]
        bucket = lsh_hash(band_sig)
        buckets_array[i][bucket] = set()

    for uid,sig in user_sigs.iteritems():
      for i in range(num_bands):
        si = i*B
        band_sig = sig[si:si+B]
        bucket = lsh_hash(band_sig)
        buckets_array[i][bucket].add(uid)
  else:
    target_user_set = set(target_users)
    for uid,sig in user_sigs.iteritems():
      if uid in target_user_set:
        for i in range(num_bands):
          si = i*B
          band_sig = sig[si:si+B]
          bucket = lsh_hash(band_sig)
          buckets_array[i][bucket] = set()

    for uid,sig in user_sigs.iteritems():
      if uid in target_user_set:
        for i in range(num_bands):
          si = i*B
          band_sig = sig[si:si+B]
          bucket = lsh_hash(band_sig)
          buckets_array[i][bucket].add(uid)

  return buckets_array

def lsh_query(buckets, query_sig):
  similar_arr = lsh_query_return_array(buckets, query_sig)
  return set.union(*similar_arr)

def lsh_query_return_array(buckets, query_sig, B=band_width):
  similar_arr = [set() for i in range(len(buckets))]
  for i in range(len(buckets)): # iterate different bands
    si = i*B
    query_band_sig = query_sig[si:si+B]
    query_band_bucket = lsh_hash(query_band_sig)
    similar_arr[i] = buckets[i][query_band_bucket]
  return similar_arr

def lsh_query_optimize(buckets3, buckets4, buckets5, query_sig):
  print('    Start optimizing the nearest neighbours...')
  result5 = lsh_query_return_array(buckets5, query_sig, B=5)
  set5 = set.union(*result5)
  length5 = len(set5)
  if length5 <= neighbours_upper_bound and length5 >= neighbours_lower_bound:
    print('    number of neighbours fall into the correct range, return directly.')
    return set5
  else:
    print('    number of bandwidth 5: %d' % length5)
    if length5 > neighbours_upper_bound: # find the ones that have >1 shared buckets
      print('    too many neighbours at 5, filter them down: ' + str(length5))
      return most_frequent_items(result5, 5)
      #return random.sample(set5, max_neighbours_sim)
    else: # find the ones from bucket3 and bucket4
      print('    too few neighbours at 5, get from 4')
      result4 = lsh_query_return_array(buckets4, query_sig, B=4)
      set4 = set.union(*result4)
      print('    number of bandwidth 4: %d' % len(set4))
      if len(set4.union(set5)) >= neighbours_lower_bound and len(set4.union(set5)) <= neighbours_upper_bound:
        return set4.union(set5)
      else:
        if len(set4.union(set5)) > neighbours_upper_bound:
          print('    too many neighbours at 4 U 5, filter them down: ' + str(len(set4.union(set5))))
          return most_frequent_items(result4, 4)
          #return set5.union(random.sample(set4.difference(set5), max_neighbours_sim - length5))
        else:
          print('    too few neighbours at 4 U 5, get from 3')
          result3 = lsh_query_return_array(buckets3, query_sig, B=3)
          set3 = set.union(*result3)
          print ('    number of bandwidth 3: %d' % len(set3))
          if len(set.union(set3, set4, set5)) > max_neighbours_sim:
            print('    too many neighbours at 3 U 4 U 5, random sample them down')
            return set.union(set5, set4, random.sample((set3.difference(set4)).difference(set5), max_neighbours_sim - len(set5.union(set4))))
          else:
            return set.union(set3, set4, set5)

def most_frequent_items(similar_arr, bandwidth):
  num_set = sig_dim/bandwidth
  counter = 0
  result = set()
  neighbour_dict = {}
  for i in range (0, num_set):
    for item in similar_arr[i]:
      if item in neighbour_dict:
        neighbour_dict[item] = neighbour_dict[item] + 1
      else:
        neighbour_dict[item] = 1
  for i in range (0, num_set):
    for key, value in neighbour_dict.iteritems():
      if value == num_set - i:
        counter = counter + 1
        result.add(key)
        if counter >= max_neighbours_sim:
          break
    if counter >= max_neighbours_sim:
      break

  return result

# def lsh_query(user_sigs, query_uid):
#   query_sig = user_sigs[query_uid]
#   num_bands = sig_dim / band_width
#   buckets_array = [{} for i in range(num_bands)]
#   similar_arr = [set() for i in range(num_bands)]
#   for uid,sig in user_sigs.iteritems():
#     for i in range(num_bands):
#       si = i*band_width

#       query_band_sig = query_sig[si:si+band_width]
#       query_band_bucket = lsh_hash(query_band_sig)

#       band_sig = sig[si:si+band_width]
#       band_bucket = lsh_hash(band_sig)

#       if band_bucket == query_band_bucket:
#         similar_arr[i].add(uid)
#   return set.union(*similar_arr)

def lsh_main(query_uid):
  #min_hash()   #uncomment this if need to generate the new signature file
  start_time = time.time()
  user_sig_dict = read_dict_from_file(signature_path)
  print ("Signature File Load Time: %.2fs" % (time.time()-start_time))

  start_time = time.time()
  buckets = lsh(user_sig_dict)
  print ("LSH Buckets Creation Time: %.2fs" % (time.time()-start_time))

  query_sig = user_sig_dict[query_uid]
  start_time = time.time()
  result = lsh_query(buckets, query_sig)
  print ("Similarity Query Time: %.5fs" % (time.time()-start_time))

  print len(result)
  print result
  return result

