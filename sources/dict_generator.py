import const
import numpy as np

'''
return a dict of mapping from item id to a distinct number
'''
def get_item_dict():
  item_dict = {} # item_id -> item idx
  with open(const.item_path) as fs:
    contents = fs.readlines()
  idx = 0
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    item_id = int(segs[0])
    item_dict[item_id] = idx # assume no duplicate item_ids
    idx += 1
  return item_dict

'''
return a dict of mapping from follower id to a list of user id's mapping
that it is following
'''
def get_follow_dict():
  item_dict = get_item_dict()
  follow_dict = {} # user_id -> array of item indices

  itemKeySet = set(item_dict.keys())

  with open(const.sns_path) as fs:
    contents = fs.readlines()
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    follower_id = int(segs[0])
    follow_dict[follower_id] = set()
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    follower_id = int(segs[0])
    followee_id = int(segs[1])
    if followee_id in itemKeySet:
	    follow_dict[follower_id].add(item_dict[followee_id])
  return follow_dict
  
def get_actions_dict(item_dict):
  actions_dict = {}
  itemKeySet = set(item_dict.keys())
  with open(const.action_path) as fs:
  	 contents = fs.readlines()
     
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    srcUser = int(segs[0])
    destUser = int(segs[1])
    if destUser in itemKeySet:
      actions_dict[srcUser] = []
    
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    srcUser = int(segs[0])
    destUser = int(segs[1])
    if destUser in itemKeySet:
      atActions = int(segs[2])
      retweetActions = int(segs[3])
      commentActions = int(segs[4])
      actions_dict[srcUser].append((item_dict[destUser],atActions,retweetActions,commentActions))
  
  return actions_dict
  
def get_keywords_dict():
  keywords_dict = {}
  with open(const.keyword_path) as fs:
    contents = fs.readlines()
    
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    uid = int(segs[0])
    keywords_dict[uid] = []
  
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    uid = int(segs[0])
    words = segs[1].split(';')
    for w in words:
      w = w.strip()
      wid = float(w.split(':')[0])
      wt = float(w.split(':')[1])
      keywords_dict[uid].append((wid,wt))
  
  return keywords_dict
  
# def get_keywords_dict(K,neighbours,userIdxDict):
  # keywords_dict = {}
  # with open(const.keyword_path) as fs:
    # contents = fs.readlines()
  # for uid in neighbours:
    # keywords_dict[userIdxDict[uid]] = []
  # for line in contents:
    # line = line.strip()
    # segs = line.split('\t')
    # uid = int(segs[0])
    # if uid in neighbours:
      # words = segs[1].split(';')
      # for w in words:
        # w = w.strip()
        # wt = float(w.split(':')[1])
        # keywords_dict[userIdxDict[uid]].append((np.random.rand(K),wt))
  # return keywords_dict

if __name__ == '__main__':
  follow_dict = get_follow_dict()
  f = open('user_follows.txt','w')
  for uid,items in follow_dict.iteritems():
    line = ''+str(uid)+'\t'
    for i in sorted(list(items)):
      line += str(i)+' '
    line = line.strip() + '\n'
    f.write(line)
  f.close()
