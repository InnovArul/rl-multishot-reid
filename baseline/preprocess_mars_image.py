import os
import subprocess
import csv
import random
import glob
import numpy as np

ROOT = os.path.abspath('../mars')
output = os.path.abspath('../mars/mxnet-recs')
im2rec = '/home/arul/anaconda3/envs/mxnet-rl/lib/python3.7/site-packages/mxnet/tools/im2rec.py'

def load_split():
    train, test = [], []
    cnt = 0
    for i in range(386):
        cam_a = glob.glob('%s/multi_shot/cam_a/person_%04d/*.png' % (ROOT, i))
        cam_b = glob.glob('%s/multi_shot/cam_b/person_%04d/*.png' % (ROOT, i))
        if len(cam_a) * len(cam_b) > 0:
            cnt += 1
            if cnt > 100:
                test.append(i)
            else:
                train.append(i)
            if cnt >= 200:
                break
    return train, test

def rnd_pos(N, i):
    x = random.randint(0, N - 2)
    return x + 1 if x == i else x

def save_rec(lst, path, name):
    prefix = '%s/%s' % (path, name)
    lst_file = prefix + '.lst'

    #print lst_file, rec_file, '%s %s %s %s resize=128 quality=90' % (im2rec, lst_file, ROOT, rec_file)
    fo = csv.writer(open(lst_file, "w"), delimiter='\t', lineterminator='\n')
    for item in lst:
        fo.writerow(item)

    command = '%s "%s" "%s" --resize 128 --quality 90' % (im2rec, prefix, ROOT) 
    print(command)
    os.system(command)

def save_train(f, is_valid=False):
    plst, nlst, cnt, N, pool = [], [], 0, len(f), [_ for _ in range(len(f))]
    for _ in range(100000 if not is_valid else 2000):
        ts = random.sample(pool, 96)
        ns, ps = ts[:64], ts[64:]
        for r in range(32):
            i, x, y = ps[r], ns[r + r], ns[r + r + 1]
            p1c = random.randint(0, len(f[i]) - 1)
            p2c = rnd_pos(len(f[i]), p1c)
            p1 = (cnt, i, f[i][p1c][random.randint(0, len(f[i][p1c]) - 1)])
            p2 = (cnt + 1, i, f[i][p2c][random.randint(0, len(f[i][p2c]) - 1)])
            n1c = random.randint(0, len(f[x]) - 1)
            n2c = random.randint(0, len(f[y]) - 1)
            n1 = (cnt, x, f[x][n1c][random.randint(0, len(f[x][n1c]) - 1)])
            n2 = (cnt + 1, y, f[y][n2c][random.randint(0, len(f[y][n2c]) - 1)])
            cnt += 2
            plst.append(p1)
            plst.append(p2)
            nlst.append(n1)
            nlst.append(n2)
    save_rec(plst, output, 'image_' + ('valid' if is_valid else 'train') + '_even')
    save_rec(nlst, output, 'image_' + ('valid' if is_valid else 'train') + '_rand')

def gen_train():
    pool = []
    for i in range(1500):
        images = glob.glob('%s/bbox_train/%04d/*.jpg' % (ROOT, i))
        f = dict()
        for k in images:
            name = k.split('/')[-1]
            ct = name[4:6]
            if not ct in f:
                f[ct] = []
            f[ct].append(k)
        g = []
        for x in f:
            if len(f[x]) > 1:
                g.append(f[x])
        if len(g) <= 1:
            continue
        pool.append(g)

    save_train(pool)
    save_train(pool, is_valid=True)

def naive_lst(dataset):
    lst_file = open('%s/MARS-evaluation/info/%s_name.txt' % (ROOT, dataset))
    lst, cnt = [], 0
    for line in lst_file:
        s = line.strip()
        lst.append((cnt, 0, '/bbox_%s/%s/%s' % (dataset, s[:4], s)))
        cnt += 1
    save_rec(lst, output, 'eval_' + dataset)

if __name__ == '__main__':
    #naive_lst('train')
    #naive_lst('test')
    gen_train()
