
from math import log
import operator                                                                 


class ActiveLearner:
    def __init__(self, matcher, batch_size, max_iters, gold_file):
        self.matcher = matcher
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.gold_pairs = load_gold_pairs(gold_file)

    def learn(self, candset, candset_key_attr, candset_l_key_attr, 
              candset_r_key_attr):
        unlabeled_pairs = candset.set_index(candset_key_attr) 

        feature_attrs = list(unlabeled_pairs.columns)
        feature_attrs.remove(candset_l_key_attr)
        feature_attrs.remove(candset_r_key_attr)
               
        # randomly select first batch of pairs to label 
        first_batch = unlabeled_pairs.sample(self.batch_size)

        # get labels for first batch
        labeled_pairs = self.label_pairs(first_batch, candset_l_key_attr, 
                                    candset_r_key_attr)

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
            labeled_current_batch = self.label_pairs(current_batch, 
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
        
        entropies = {}
        for i in xrange(len(probabilities)):
            entropies[i] = self._compute_entropy(probabilities[i])

        batch_idxs = []
        for pair_entropy in sorted(entropies.items(), 
                                   key=operator.itemgetter(1), reverse=True):   
            if len(batch_idxs) == self.batch_size:
                break
            batch_idxs.append(pair_entropy[0])

        return unlabeled_pairs.iloc[batch_idxs]

    def _compute_entropy(self, arr):
        entropy = 0
        for prob in arr:
            if prob > 0:
                entropy += prob * log(prob)
#        if arr[0] > 0:
#            entropy += arr[0] * log(arr[0])
#        if arr[1] > 0:
#            entropy += arr[1] * log(arr[1])
        return entropy    

    def label_pairs(self, to_be_labeled_pairs, l_key_attr, r_key_attr):
        labels = (to_be_labeled_pairs[l_key_attr].astype(str) + ',' + 
            to_be_labeled_pairs[r_key_attr].astype(str)).apply(
                                        lambda val: self.gold_pairs.get(val, 0))
        to_be_labeled_pairs['label'] = labels
        return to_be_labeled_pairs

def load_gold_pairs(gold_file):
    gold_pairs = {}
    file_handle = open(gold_file, 'r')
    for line in file_handle:
        gold_pairs[line.strip()] = 1
    return gold_pairs
    
