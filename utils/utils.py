
import os
import random 
import numpy as np
import time


def make_dataset(directory_ld, class_to_idx, directory_audio= None):
    instances = []
    directory = os.path.expanduser(directory_ld)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path_ld = os.path.join(root, fname)
                if directory_audio is not None:
                    path_audio = os.path.join(directory_audio,root.split("/")[-1], "03"+fname[2:-3]+"pkl")
                    item = path_ld,class_index, path_audio
                else:
                    item = path_ld,class_index
                    
                instances.append(item)
    return instances

def find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def weighted_acc(truths,preds,  verbose=False):
    # preds = preds.view(-1)
    # truths = truths.view(-1)

    total = len(preds)
    tp = 0
    tn = 0
    p = 0
    n = 0
    for i in range(total):
        if truths[i] == 0:
            n += 1
            if preds[i] == 0:
                tn += 1
        elif truths[i] == 1:
            p += 1
            if preds[i] == 1:
                tp += 1

    w_acc = (tp * n / p + tn) / (2 * n)

    if verbose:
        fp = n - tn
        fn = p - tp
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        print('TP=', tp, 'TN=', tn, 'FP=', fp, 'FN=', fn, 'P=', p, 'N', n, 'Recall', recall, "f1", f1)

    return w_acc


def split_dataset(path=None,path_audio=None, perc=0.9,actor_split=False,seed=42):

    if path is None:
        msg = "Path to the dataset folder is needed"
        raise RuntimeError(msg)

    instances = make_dataset(path,find_classes(path)[1],directory_audio=path_audio)
    random.seed(seed)

    if actor_split==False:
        class_sep = [ [] for _ in find_classes(path)[0]]
        for k in instances:
            class_sep[k[1]].append(k)

        perc = perc
        cl_train = []
        full_train = []

        cl_test = []
        cl_valid = []
        for cl in class_sep:
            n_data = len(cl)
            len_train = int(n_data*perc)
            len_valid = int(len_train*0.1)
            len_test = n_data - len_train
            train_samples = random.sample(cl,len_train)
            
            full_train += train_samples
            valid_samples = random.sample(train_samples, len_valid)
            cl_valid += valid_samples
            train_samples = list(set(train_samples) - set(valid_samples))
            cl_train += train_samples

        cl_test = list(set(instances).difference(full_train))
        return cl_test, cl_train, cl_valid
    else:

        l_actor = [ [] for i in range(24)]
        for ist in instances:
            l_actor[int(ist[0].split(".")[0].split("-")[-1])-1].append(ist)
        random.shuffle(l_actor)
        train_split_n = int(len(l_actor)*perc)
        split_train =  int(train_split_n * 0.9)
        

        actor_train = l_actor[:split_train]
        actor_valid = l_actor[split_train:train_split_n]
        actor_test = l_actor[train_split_n:]


        actor_train = [item for sublist in actor_train for item in sublist]
        actor_test = [item for sublist in actor_test for item in sublist]
        actor_valid = [item for sublist in actor_valid for item in sublist]

        return actor_test, actor_train, actor_valid

def time_print(label=None, start_time=None):
    end_time = time.time()
    print(f"{label} --- {start_time - end_time}")