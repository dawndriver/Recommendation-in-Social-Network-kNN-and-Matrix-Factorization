sns_path = 'data/user_sns.txt'
item_path = 'data/item.txt'
log_path = 'data/rec_log_train.txt'
log_out = 'data/logs/log'
log_user = 'data/log_user.txt'
sample_log_user = 'data/sample_log_user.txt'
action_path = 'data/user_action.txt'
keyword_path = 'data/user_key_word.txt'
profile_path = 'data/user_profile.txt'
follow_path = 'data/user_follows.txt'
signature_path = 'data/user_signature_d20.txt'

num_items = 6095
minhash_prime = 6091
sig_dim = 20
band_width = 5

max_neighbours_sim = 50
neighbours_upper_bound = 50    # above this may look for neighbours in more restricted way or using higher bandwidth LSH buckets
neighbours_lower_bound = 10     # below this may look for neighbours in lower bandwidth LSH buckets

def read_dict_from_file(file_path):
  sig_dict = {}
  with open(file_path) as fs:
    contents = fs.readlines()
  for line in contents:
    line = line.strip()
    segs = line.split('\t')
    uid = int(segs[0])
    if len(segs) < 2:
      items_str = []
    else:
      items_str = segs[1].split(' ')
    items = [int(i) for i in items_str]
    sig_dict[uid] = items
  return sig_dict

def write_dict_into_file(dictionary, file_path):
  with open(file_path, 'w+') as fs:
    for key, value in dictionary.iteritems():
      result = ''
      result = str(key) + '\t'
      for i in value:
        result += str(i)
        result += ' '
      result = result.strip()
      result += '\n'
      fs.write(result)
    return
