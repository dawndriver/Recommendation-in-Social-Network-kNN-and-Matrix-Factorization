from itertools import islice
from const import *

follower_set = set()
followee_set = set()
follow_dict = {}
user_profile_dict = {}
item_dict = {}

def parse_user_sns ():
  with open(sns_path) as fs:
    contents = fs.readlines()
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    follower_id = segs[0]
    followee_id = segs[1]
    follower_set.add(follower_id)
    followee_set.add(followee_id)
  return

def parse_user_profile ():
  with open(profile_path) as fs:
    contents = fs.readlines()
    for line in contents:
      line = line.strip()
      segs = line.split('\t')
      user_profile_dict[segs[0]] = segs
  return user_profile_dict

def parse_item ():
  with open(item_path) as fs:
    contents = fs.readlines()
    for line in contents:
      line = line.strip()
      segs = line.split('\t')
      item_dict[segs[0]] = segs
  return item_dict

'''
from here they are all test functions
'''
def parse_user_profile_test ():
  user_profile_dict_result =  parse_user_profile()
  print len(user_profile_dict_result)
  user_profile_dict_result_10 = {}
  user_profile_dict_result_10 = list(islice(user_profile_dict_result, 10))
  for x in user_profile_dict_result_10:
    print(x)
    print user_profile_dict_result[x]
  return

def check_if_all_items_in_profile():
  counter_match = 0
  item_dict = parse_item()
  user_profile_dict = parse_user_profile()
  for key in item_dict:
    if user_profile_dict[key] is not None:
      counter_match = counter_match + 1
  print('Total items: ' + str(len(item_dict)))
  print('Total profiles: ' + str(len(user_profile_dict)))
  print('Matches: ' + str(counter_match))

  return

def check_if_all_user_sns_in_profile():
  parse_user_sns()
  parse_user_profile()
  counter_follower_match = 0
  counter_followee_match = 0
  for item in follower_set:
    if user_profile_dict[item] is not None:
      counter_follower_match = counter_follower_match + 1
  for item in followee_set:
    if user_profile_dict[item] is not None:
      counter_followee_match = counter_followee_match + 1
  print('Total followers: ' + str(len(follower_set)))
  print('Total followee: ' + str(len(followee_set)))
  print('Matches followers: ' + str(counter_follower_match))
  print('Matches followees: ' + str(counter_followee_match))

if __name__ == "__main__":
  check_if_all_user_sns_in_profile()

