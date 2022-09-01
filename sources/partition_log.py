from const import *
import random
import itertools

def partition_rand():
  shuffle = True
  step = 7000000
  with open(log_path) as fs:
    i = 1
    while True:
      next_n_lines = list(itertools.islice(fs, step))
      if not next_n_lines:
        break
      if shuffle:
        random.shuffle(next_n_lines)
      with open(log_out+str(i)+'.txt','w+') as fw:
        fw.writelines(next_n_lines)     
      i = i + 1

def transform_per_user():
  with open(log_path) as fs:
    contents = fs.readlines()
  tempD = {}
  for l in contents:
    l = l.strip()
    segs = l.split('\t')
    tempD[int(segs[0])] = 1
  d = {}
  for uid in tempD.keys():
    d[uid] = []
  for l in contents:
    l = l.strip()
    segs = l.split('\t')
    userId = int(segs[0])
    itemId = int(segs[1])
    result = int(segs[2])
    d[userId].append((itemId,result))
  with open('log_user.txt','w+') as fs:
    for uid in d.keys():
      s = ''
      for (itemId,result) in d[uid]:
        s = s + '%d,%d ' % (itemId,result)
      fs.write(str(uid) + '\t' + s + '\n')
    
def sample_data(infile, outfile, percentage):
  with open(infile) as fs:
    contents = fs.readlines()
  sample = random.sample(contents,int(percentage*len(contents)))
  with open(outfile,'w+') as fw:
    fw.writelines(sample)
    
def sample_log_user(n):
  with open(log_user) as fs:
    contents = fs.readlines()
  random.shuffle(contents)
  lines = random.sample(contents,n)
  with open('data/sample_log_user.txt', 'w+') as fw:
    fw.writelines(lines)
    
if __name__ == '__main__':
  sample_log_user(0.1)