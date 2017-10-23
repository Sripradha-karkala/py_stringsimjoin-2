
from math import log
import operator
from py_stringsimjoin.utils.generic_helper import remove_exclude_attr

class ActiveLearner:
    """
    A class which allows to match entities by actively querying the labels
    of unlabeled instances using Pool-based active learning.

    Args:
        Model (model): Scikit-Learn Model to learn
        example_selector: example selector to query informative examples
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
        self.max_iters = max_iters
        self.labeler = labeler
        self.example_selector = example_selector

    def learn(self, unlabeled_dataset, seed, exclude_attrs=None, context=None, label_attr='label'):

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

        #Add validation to the input tables

        # find the attributes to be used as features
        feature_attrs = list(unlabeled_dataset.columns)

        # Remove any excluded attributes
        feature_attrs = remove_exclude_attr(feature_attrs, exclude_attrs, unlabeled_dataset)

        labeled_pairs = seed

        unlabeled_pairs = unlabeled_pairs.drop(labeled_pairs.index)

        while self.max_iters > 0:
            # train matcher using the current set of labeled pairs
            self.model = self.model.fit(labeled_pairs[feature_attrs].values,
                                            labeled_pairs[label_attr].values)

            # select next batch to label
            current_batch = self.example_selector.select_examples(unlabeled_pairs,
                                                    self.model, exclude_attr, self.batch_size)

            # get labels for current batch

             # label the selected examples
            labeled_examples = self.labeler.label(selected_examples, context,
                                                  label_attr)

            # remove labeled pairs from the unlabeled pairs
            unlabeled_pairs = unlabeled_pairs.drop(labeled_examples.index)

            # append the current batch of labeled pairs to the previous
            labeled_pairs = labeled_pairs.append(labeled_examples)

            self.max_iters-= self.max_iters;
        return labeled_pairs
