from math import log
import operator
from py_stringsimjoin.utils.generic_helper import remove_exclude_attr
from py_stringsimjoin.utils.validation import *
from py_stringsimjoin.labeler.labeler import Labeler
from py_stringsimjoin.exampleselector.example_selector import ExampleSelector

class ActiveLearner:
    """
    A class which allows to match entities by actively querying the labels
    of unlabeled instances using Pool-based active learning.

    Args:
        Model (model): Scikit-Learn Model to learn
        Example_selector(Example Selector): example selector to query informative examples
        labeler (Labeler): A labeler which fetches labels from the oracle/human
        batch_size (int): The number of examples to be labeled per iteration
        num_iters (int): Number of iterations to run the active learner

    Attributes:
        model (Model): An attribute to store the Scikit-Learn Model
        example_selector (ExampleSelector): An attribute to store the example selector.
        labeler (Labeler): An attribute to store the labeler passed in arguments
        batch_size (int): An attribute to store the batch_size
        num_iters (int): An attribute to store the number of iterations
    """

    def __init__(self, model, example_selector, labeler, batch_size, num_iters):
        self.model = model
        self.batch_size = batch_size
        self.max_iters = num_iters
        self.labeler = labeler
        self.example_selector = example_selector
    
    def _generate_labelled_data(self, seed, unlabeled_dataset):
        #Generate the first set of labeled data set using the seed file
        seed_pairs = {}
	
        for seed_rows in seed.itertuples(index=False):
            seed_pairs[str(seed_rows[0]) + ',' + str(seed_rows[1])] = int(seed_rows[2])

        first_batch = unlabeled_dataset[unlabeled_dataset.apply(lambda row: seed_pairs.get(str(row['l_id'].astype(int))+','+str(row['r_id'].astype(int))) != None, 1)].copy()
	
        labels = (first_batch['l_id'].astype(str) +',' + first_batch['r_id'].astype(str)) \
                  .apply(lambda value : 
                         seed_pairs.get(value, 0))
        first_batch['label'] = labels
        return first_batch

    def learn(self, unlabeled_dataset, seed, exclude_attrs=[], context=None, label_attr='label'):

        """
        Performs the Active Learning Loop to help learn the model by querying
        the labels of the instances

        Args:
        unlabeled_dataset (DataFrame): A Dataframe containing unlabeled
                       examples

        seed (DataFrame): A Dataframe containing initial labeled examples
                  which is used to learn the initial model

        exclude_attrs (list): A list of attributes to be excluded while
                  fitting the model (Defaults to None)

        context (dictionary): A dictionary containing all the necessary
                    context for the labeling function

        label_attr (string): A string indicating the name of the label
                    column in the labeled dataset. Defaults to label

        Returns:
            A learned model
        """
        #validate input tables
        validate_input_table(unlabeled_dataset, 'unlabeled dataset')
        validate_input_table(seed, 'seed')
        
        #validate labeler
        if not isinstance(self.labeler, Labeler):
            raise TypeError(self.labeler + ' is not an object of labeler class')
            
        #validate example selector
        if not isinstance(self.example_selector, ExampleSelector):
            raise TypeError(self.example_selector + ' is not an object of example selector ')

        # find the attributes to be used as features
        feature_attrs = list(unlabeled_dataset.columns)

        # Remove any excluded attributes
        feature_attrs = remove_exclude_attr(feature_attrs, exclude_attrs, unlabeled_dataset)
        
        #Generate the first labelled pairs from seed data
        labeled_pairs = self._generate_labelled_data(seed, unlabeled_dataset)
        # Check with Paul, we probably do not need the above function
        # labeled_pairs = seed
        unlabeled_pairs = unlabeled_dataset.drop(labeled_pairs.index)
        i = 0

        while i < self.max_iters:
            # train matcher using the current set of labeled pairs
            self.model = self.model.fit(labeled_pairs[feature_attrs].values,
                                            labeled_pairs[label_attr].values)

            # select next batch to label
            selected_examples = self.example_selector.select_examples(unlabeled_pairs,
                                                    self.model, exclude_attrs, self.batch_size)

            # get labels for current batch
            
            
            # label the selected examples
            labeled_examples = self.labeler.label(selected_examples, context,
                                                  label_attr)

            # remove labeled pairs from the unlabeled pairs
            unlabeled_pairs = unlabeled_pairs.drop(labeled_examples.index)

            # append the current batch of labeled pairs to the previous
            labeled_pairs = labeled_pairs.append(labeled_examples)

            i = i + 1
        return labeled_pairs
		
