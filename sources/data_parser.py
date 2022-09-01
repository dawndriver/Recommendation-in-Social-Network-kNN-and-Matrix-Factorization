from const import *

follower_set = set()
followee_set = set()
follow_dict = {}

train_uid_set = set()
train_item_set = set()
item_id_set = set()

item_dict = {} # item_id -> idx

'''
Go through sns file to save all:
    followers into follower_set
    followees into followee_set
    follow_dict to store a dic of followers and list of followees
'''
def parse_user_sns ():
  with open(sns_path) as fs:
    contents = fs.readlines()
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    follower_id = int(segs[0])
    followee_id = int(segs[1])
    if follower_id not in follower_set:
      follower_set.add(follower_id)
    if followee_id not in followee_set:
      followee_set.add(followee_id)
    if follow_dict.has_key(follower_id):
      follow_dict[follower_id].append(followee_id)
    else:
      follow_dict[follower_id] = []

'''
Go through the item file and store the mapping of itemId and the number we give it
'''
def prepare_item_dict():
  with open(item_path) as fs:
    contents = fs.readlines()
  idx = 0
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    item_id = int(segs[0])
    item_dict[item_id] = idx # assume no duplicate item_ids
    idx += 1

def parse_items ():
  with open(item_path) as fs:
    contents = fs.readlines()
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    item_id = int(segs[0])
    item_id_set.add(item_id)

def parse_train ():
  with open(train_path) as fs:
    contents = fs.readlines()
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    user_id = int(segs[0])
    item_id = int(segs[1])
    if user_id not in train_uid_set:
      train_uid_set.add(user_id)
    if item_id not in train_item_set:
      train_item_set.add(item_id)

#parse_user_sns()
#print len(follower_set)
#print len(followee_set)
#parse_items()
#print len(item_id_set)
#print len(item_id_set.intersection(followee_set))
