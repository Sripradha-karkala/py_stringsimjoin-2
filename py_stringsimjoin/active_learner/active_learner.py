
from math import log
import operator                                                                 

from py_stringsimjoin.sampler.weighted_random_sampler import \
                                                        WeightedRandomSampler


class ActiveLearner:
    def __init__(self, matcher, batch_size, max_iters, gold_file, seed):
        self.matcher = matcher
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.gold_pairs = load_gold_pairs(gold_file)
        self.seed_pairs = load_seed_pairs(seed)

    def learn(self, candset, candset_key_attr, candset_l_key_attr, 
              candset_r_key_attr):
        unlabeled_pairs = candset.set_index(candset_key_attr) 
        unlabeled_pairs[candset_l_key_attr] = unlabeled_pairs[candset_l_key_attr].astype(str)
        unlabeled_pairs[candset_r_key_attr] = unlabeled_pairs[candset_r_key_attr].astype(str)
        feature_attrs = list(unlabeled_pairs.columns)
        feature_attrs.remove(candset_l_key_attr)
        feature_attrs.remove(candset_r_key_attr)
               
        # randomly select first batch of pairs to label 
#        first_batch = unlabeled_pairs.sample(self.batch_size)    
        first_batch = self._get_first_batch(unlabeled_pairs, candset_l_key_attr,
                                            candset_r_key_attr)
        print first_batch
        # get labels for first batch
        labeled_pairs = self._label_pairs(first_batch, candset_l_key_attr, 
                                          candset_r_key_attr)
        print labeled_pairs
        # remove labeled pairs from the unlabeled pairs
        unlabeled_pairs = unlabeled_pairs.drop(labeled_pairs.index)

        current_iter = 0
        
        while current_iter < self.max_iters:
            # train matcher using the current set of labeled pairs
            self.matcher = self.matcher.fit(labeled_pairs[feature_attrs].values,
                                            labeled_pairs['label'].values)

            # select next batch to label
            print('Selecting next batch...')
            current_batch = self._select_next_batch(unlabeled_pairs, 
                                                    feature_attrs)

            # get labels for current batch
            print('Collecting labels...')
            labeled_current_batch = self._label_pairs(current_batch, 
                                        candset_l_key_attr, candset_r_key_attr)

            # remove labeled pairs from the unlabeled pairs                         
            unlabeled_pairs = unlabeled_pairs.drop(labeled_current_batch.index)     
           
            # append the current batch of labeled pairs to the previous 
            # labeled pairs
            labeled_pairs = labeled_pairs.append(labeled_current_batch)

            current_iter += 1
            print 'Iteration :', current_iter

    def _select_next_batch(self, unlabeled_pairs, feature_attrs):
        probabilities = self.matcher.predict_proba(
                            unlabeled_pairs[feature_attrs].values)
        print 'computing entropy'        
        entropies = {}
        for i in xrange(len(probabilities)):
            entropy = self._compute_entropy(probabilities[i])
            if entropy > 0:
                entropies[i] = entropy

        print 'sorting'
        top_k_pairs = sorted(entropies.items(),                           
            key=operator.itemgetter(1), reverse=True)[:min(100, len(entropies))]

        weights = map(lambda val: val[1], top_k_pairs)
        selected_pairs = map(lambda val: False, top_k_pairs)
        print 'sampling'
        wrs = WeightedRandomSampler(weights)
        next_batch_idxs = []
        print weights
        while len(next_batch_idxs) < self.batch_size and len(next_batch_idxs) < len(top_k_pairs):
            pair_idx = wrs.next()
#            print pair_idx, len(next_batch_idxs)
            if selected_pairs[pair_idx]:
                continue
            selected_pairs[pair_idx] = True
            next_batch_idxs.append(top_k_pairs[pair_idx][0])
#        sample = wsample(weights, self.batch_size)
#        next_batch_idxs = map(lambda pair_idx: top_k_pairs[pair_idx][0], sample)      
        return unlabeled_pairs.iloc[next_batch_idxs]

    def _compute_entropy(self, arr):
        entropy = 0
        for prob in arr:
            if prob > 0:
                entropy += prob * log(prob)
#        if arr[0] > 0:
#            entropy += arr[0] * log(arr[0])
#        if arr[1] > 0:
#            entropy += arr[1] * log(arr[1])
        if entropy != 0:
            entropy = entropy * -1
        return entropy    

    def _label_pairs(self, to_be_labeled_pairs, l_key_attr, r_key_attr):
        labels = (to_be_labeled_pairs[l_key_attr].astype(str) + ',' + 
            to_be_labeled_pairs[r_key_attr].astype(str)).apply(
                                        lambda val: self.gold_pairs.get(val, 0))
        to_be_labeled_pairs['label'] = labels
        return to_be_labeled_pairs

    def _get_first_batch(self, unlabeled_pairs, l_key_attr, r_key_attr):
        print (unlabeled_pairs.columns)
        print l_key_attr, r_key_attr
        return unlabeled_pairs[unlabeled_pairs.apply(lambda row: 
            self.seed_pairs.get(str(row[l_key_attr]) + ',' + 
                                str(row[r_key_attr])) != None, 1)].copy()


def load_gold_pairs(gold_file):
    gold_pairs = {}
    file_handle = open(gold_file, 'r')
    for line in file_handle:
        gold_pairs[line.strip()] = 1
    file_handle.close()
    return gold_pairs


def load_seed_pairs(seed):                                                 
    seed_pairs = {}
    for seed_pair_row in seed.itertuples(index=False):
        seed_pairs[str(seed_pair_row[0]) + ',' + 
                   str(seed_pair_row[1])] = int(seed_pair_row[2])                                                        
#    file_handle = open(seed_file, 'r')                                          
#    for line in file_handle:
#        fields = line.strip().split(',')                                                    
#        seed_pairs[fields[0]+','+fields[1]] = int(fields[2])
#    file_handle.close()                                            
    return seed_pairs   

def wsample(weights, sample_size):
    items = range(0, len(weights))
    sample = []
    cum_weight = 0.0
    for i in range(0, sample_size):
        sample.append(i)
        cum_weight += weights[i]
    for i in range(sample_size, len(weights)):
        prob = weights[i] / cum_weight
        if random.random() <= prob:
            sample[random.randint(0, sample_size - 1)] = items[i]
        cum_weight += weights[i]
    return sample
        
    
