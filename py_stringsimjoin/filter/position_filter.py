# Position Filter

from joblib import delayed, Parallel
from six import iteritems
from six.moves import xrange
import pandas as pd
import pyprind

from py_stringsimjoin.filter.filter import Filter
from py_stringsimjoin.filter.filter_utils import get_overlap_threshold, \
    get_prefix_length, get_size_lower_bound, get_size_upper_bound
from py_stringsimjoin.index.position_index import PositionIndex
from py_stringsimjoin.utils.generic_helper import convert_dataframe_to_list, \
    find_output_attribute_indices, get_attrs_to_project, \
    get_num_processes_to_launch, get_output_header_from_tables, \
    get_output_row_from_tables, remove_redundant_attrs, split_table
from py_stringsimjoin.utils.missing_value_handler import \
    get_pairs_with_missing_value
from py_stringsimjoin.utils.token_ordering import gen_token_ordering_for_lists,\
    gen_token_ordering_for_tables, order_using_token_ordering
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_key_attr, validate_input_table, validate_threshold, \
    validate_tokenizer_for_sim_measure, validate_output_attrs, \
    validate_sim_measure_type


class PositionFilter(Filter):
    """Finds candidate matching pairs of strings using position filtering 
    technique.

    For similarity measures such as cosine, Dice, Jaccard and overlap, the 
    filter finds candidate string pairs that may have similarity score greater 
    than or equal to the input threshold. Where as for distance measure such as 
    edit distance, the filter finds candidate string pairs that may have 
    distance score less than or equal to the threshold.

    To know about position filtering, refer the `string matching chapter 
    <http://pages.cs.wisc.edu/~anhai/py_stringmatching/dibook-string-matching.pdf>`_ 
    of the "Principles of Data Integration" book.

    Args:
        tokenizer (Tokenizer object): tokenizer to be used.
        sim_measure_type (string): similarity measure type. Supported types are 
            'JACCARD', 'COSINE', 'DICE', 'OVERLAP' and 'EDIT_DISTANCE'.
        threshold (float): threshold to be used by the filter.
        allow_empty (boolean): A flag to indicate whether pairs in which both   
            strings are tokenized into an empty set of tokens should            
            survive the filter (defaults to True). This flag is not valid for   
            measures such as 'OVERLAP' and 'EDIT_DISTANCE'.                     
        allow_missing (boolean): A flag to indicate whether pairs containing    
            missing value should survive the filter (defaults to False).

    Attributes:
        tokenizer (Tokenizer object): An attribute to store the tokenizer.
        sim_measure_type (string): An attribute to store the similarity measure 
            type.
        threshold (float): An attribute to store the threshold value.
        allow_empty (boolean): An attribute to store the value of the flag    
            allow_empty.
        allow_missing (boolean): An attribute to store the value of the flag 
            allow_missing.
    """

    def __init__(self, tokenizer, sim_measure_type, threshold,
                 allow_empty=True, allow_missing=False):
        # check if the sim_measure_type is valid                                
        validate_sim_measure_type(sim_measure_type)                             
        sim_measure_type = sim_measure_type.upper()
                                                                                
        # check if the input tokenizer is valid                                 
        validate_tokenizer_for_sim_measure(tokenizer, sim_measure_type) 

        # check if the threshold is valid
        validate_threshold(threshold, sim_measure_type)

        self.tokenizer = tokenizer
        self.sim_measure_type = sim_measure_type
        self.threshold = threshold
        self.allow_empty = allow_empty

        super(self.__class__, self).__init__(allow_missing)

    def filter_pair(self, lstring, rstring):
        """Checks if the input strings get dropped by the position filter.

        Args:
            lstring,rstring (string): input strings

        Returns:
            A flag indicating whether the string pair is dropped (boolean).
        """

        # If one of the inputs is missing, then check the allow_missing flag.
        # If it is set to True, then pass the pair. Else drop the pair.
        if pd.isnull(lstring) or pd.isnull(rstring):
            return (not self.allow_missing)

        ltokens = self.tokenizer.tokenize(lstring)
        rtokens = self.tokenizer.tokenize(rstring)

        l_num_tokens = len(ltokens)
        r_num_tokens = len(rtokens)

        if l_num_tokens == 0 and r_num_tokens == 0:
            if self.sim_measure_type == 'OVERLAP':
                return True
            elif self.sim_measure_type == 'EDIT_DISTANCE':
                return False
            else:
                return (not self.allow_empty)

        token_ordering = gen_token_ordering_for_lists([ltokens, rtokens])
        ordered_ltokens = order_using_token_ordering(ltokens, token_ordering)
        ordered_rtokens = order_using_token_ordering(rtokens, token_ordering)

        l_prefix_length = get_prefix_length(l_num_tokens,
                                            self.sim_measure_type,
                                            self.threshold,
                                            self.tokenizer) 
        r_prefix_length = get_prefix_length(r_num_tokens,
                                            self.sim_measure_type,
                                            self.threshold,
                                            self.tokenizer)

        if l_prefix_length <= 0 or r_prefix_length <= 0:
            return True
 
        l_prefix_dict = {}
        l_pos = 0
        for token in ordered_ltokens[0:l_prefix_length]:
            l_prefix_dict[token] = l_pos

        overlap_threshold = get_overlap_threshold(l_num_tokens, r_num_tokens,
                                                  self.sim_measure_type,
                                                  self.threshold,
                                                  self.tokenizer)
        current_overlap = 0
        r_pos = 0 
        for token in ordered_rtokens[0:r_prefix_length]:
            l_pos = l_prefix_dict.get(token)
            if l_pos is not None:
                overlap_upper_bound = 1 + min(l_num_tokens - l_pos - 1,
                                              r_num_tokens - r_pos - 1)
                if (current_overlap + overlap_upper_bound) < overlap_threshold:
                    return True
                current_overlap += 1
            r_pos += 1

        if current_overlap > 0:
            return False
        return True
        
    def filter_tables(self, ltable, rtable,
                      l_key_attr, r_key_attr,
                      l_filter_attr, r_filter_attr,
                      l_out_attrs=None, r_out_attrs=None,
                      l_out_prefix='l_', r_out_prefix='r_',
                      n_jobs=1, show_progress=True):
        """Finds candidate matching pairs of strings from the input tables using
        position filtering technique.

        Args:
            ltable (dataframe): left input table.

            rtable (dataframe): right input table.

            l_key_attr (string): key attribute in left table.

            r_key_attr (string): key attribute in right table.

            l_filter_attr (string): attribute in left table on which the filter 
                should be applied.                                              
                                                                                
            r_filter_attr (string): attribute in right table on which the filter
                should be applied.                                              
                                                                                
            l_out_attrs (list): list of attribute names from the left table to  
                be included in the output table (defaults to None).             
                                                                                
            r_out_attrs (list): list of attribute names from the right table to 
                be included in the output table (defaults to None).             
                                                                                
            l_out_prefix (string): prefix to be used for the attribute names    
                coming from the left table, in the output table                 
                (defaults to 'l\_').                                            
                                                                                
            r_out_prefix (string): prefix to be used for the attribute names    
                coming from the right table, in the output table                
                (defaults to 'r\_').                                            
                                                                                
            n_jobs (int): number of parallel jobs to use for the computation    
                (defaults to 1). If -1 all CPUs are used. If 1 is given, no     
                parallel computing code is used at all, which is useful for     
                debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. 
                Thus for n_jobs = -2, all CPUs but one are used. If             
                (n_cpus + 1 + n_jobs) becomes less than 1, then n_jobs is set   
                to 1.                                                           
                                                                                
            show_progress (boolean): flag to indicate whether task progress     
                should be displayed to the user (defaults to True).             
                                                                                
        Returns:                                                                
            An output table containing tuple pairs that survive the filter      
            (dataframe).
        """

        # check if the input tables are dataframes
        validate_input_table(ltable, 'left table')
        validate_input_table(rtable, 'right table')

        # check if the key attributes and filter attributes exist
        validate_attr(l_key_attr, ltable.columns,
                      'key attribute', 'left table')
        validate_attr(r_key_attr, rtable.columns,
                      'key attribute', 'right table')
        validate_attr(l_filter_attr, ltable.columns,
                      'filter attribute', 'left table')
        validate_attr(r_filter_attr, rtable.columns,
                      'filter attribute', 'right table')

        # check if the output attributes exist
        validate_output_attrs(l_out_attrs, ltable.columns,
                              r_out_attrs, rtable.columns)

        # check if the key attributes are unique and do not contain 
        # missing values
        validate_key_attr(l_key_attr, ltable, 'left table')
        validate_key_attr(r_key_attr, rtable, 'right table')

        # convert the filter attributes to string type, in case it is 
        # int or float.
        revert_l_filter_attr_type = False
        orig_l_filter_attr_type = ltable[l_filter_attr].dtype
        if (orig_l_filter_attr_type == pd.np.int64 or
            orig_l_filter_attr_type == pd.np.float64):
            ltable[l_filter_attr] = ltable[l_filter_attr].astype(str)
            revert_l_filter_attr_type = True

        revert_r_filter_attr_type = False
        orig_r_filter_attr_type = rtable[r_filter_attr].dtype
        if (orig_r_filter_attr_type == pd.np.int64 or
            orig_r_filter_attr_type == pd.np.float64):
            rtable[r_filter_attr] = rtable[r_filter_attr].astype(str)
            revert_r_filter_attr_type = True

        # remove redundant attrs from output attrs.
        l_out_attrs = remove_redundant_attrs(l_out_attrs, l_key_attr)
        r_out_attrs = remove_redundant_attrs(r_out_attrs, r_key_attr)

        # get attributes to project.  
        l_proj_attrs = get_attrs_to_project(l_out_attrs,
                                            l_key_attr, l_filter_attr)
        r_proj_attrs = get_attrs_to_project(r_out_attrs,
                                            r_key_attr, r_filter_attr)

        # do a projection on the input dataframes. Note that this doesn't create
        # a copy of the dataframes. It only creates a view on original 
        # dataframes.
        ltable_projected = ltable[l_proj_attrs]
        rtable_projected = rtable[r_proj_attrs]

        # computes the actual number of jobs to launch.
        n_jobs = min(get_num_processes_to_launch(n_jobs), len(rtable_projected))

        if n_jobs <= 1:
            output_table = _filter_tables_split(
                                           ltable_projected, rtable_projected,
                                           l_key_attr, r_key_attr,
                                           l_filter_attr, r_filter_attr,
                                           self,
                                           l_out_attrs, r_out_attrs,
                                           l_out_prefix, r_out_prefix,
                                           show_progress)
        else:
            r_splits = split_table(rtable_projected, n_jobs)
            results = Parallel(n_jobs=n_jobs)(delayed(_filter_tables_split)(
                                    ltable_projected, r_splits[job_index],
                                    l_key_attr, r_key_attr,
                                    l_filter_attr, r_filter_attr,
                                    self,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    (show_progress and (job_index==n_jobs-1)))
                                for job_index in range(n_jobs))
            output_table = pd.concat(results)

        if self.allow_missing:
            missing_pairs = get_pairs_with_missing_value(
                                            ltable_projected, rtable_projected,
                                            l_key_attr, r_key_attr,
                                            l_filter_attr, r_filter_attr,
                                            l_out_attrs, r_out_attrs,
                                            l_out_prefix, r_out_prefix,
                                            False, show_progress)
            output_table = pd.concat([output_table, missing_pairs])

        output_table.insert(0, '_id', range(0, len(output_table)))

        # revert the type of filter attributes to their original type, in case
        # it was converted to string type.
        if revert_l_filter_attr_type:
            ltable[l_filter_attr] = ltable[l_filter_attr].astype(
                                                        orig_l_filter_attr_type)

        if revert_r_filter_attr_type:
            rtable[r_filter_attr] = rtable[r_filter_attr].astype(
                                                        orig_r_filter_attr_type)

        return output_table

    def find_candidates(self, probe_tokens, position_index):
        probe_num_tokens = len(probe_tokens)
        size_lower_bound = max(get_size_lower_bound(probe_num_tokens,
                                   self.sim_measure_type, self.threshold),
                               position_index.min_length)
        size_upper_bound = min(get_size_upper_bound(probe_num_tokens,
                                   self.sim_measure_type, self.threshold),
                               position_index.max_length)

        overlap_threshold_cache = {}
        for size in xrange(size_lower_bound, size_upper_bound + 1):
            overlap_threshold_cache[size] = get_overlap_threshold(
                                                size, probe_num_tokens,
                                                self.sim_measure_type,
                                                self.threshold,
                                                self.tokenizer)

        probe_prefix_length = get_prefix_length(probe_num_tokens,
                                                self.sim_measure_type,
                                                self.threshold,
                                                self.tokenizer)

        # probe position index and find candidates
        candidate_overlap = {}
        probe_pos = 0
        for token in probe_tokens[0:probe_prefix_length]:
            for (cand, cand_pos) in position_index.probe(token):
                current_overlap = candidate_overlap.get(cand, 0)
                if current_overlap != -1:
                    cand_num_tokens = position_index.size_cache[cand]
                    if size_lower_bound <= cand_num_tokens <= size_upper_bound:
                        if (probe_num_tokens - probe_pos <=
                                cand_num_tokens - cand_pos):
                            overlap_upper_bound = probe_num_tokens - probe_pos
                        else:
                            overlap_upper_bound = cand_num_tokens - cand_pos 
                        if (current_overlap + overlap_upper_bound >=
                                overlap_threshold_cache[cand_num_tokens]):
                            candidate_overlap[cand] = current_overlap + 1
                        else:
                            candidate_overlap[cand] = -1
            probe_pos += 1

        return candidate_overlap


def _filter_tables_split(ltable, rtable,
                         l_key_attr, r_key_attr,
                         l_filter_attr, r_filter_attr,
                         position_filter,
                         l_out_attrs, r_out_attrs,
                         l_out_prefix, r_out_prefix, show_progress):
    # find column indices of key attr, filter attr and output attrs in ltable
    l_columns = list(ltable.columns.values)
    l_key_attr_index = l_columns.index(l_key_attr)
    l_filter_attr_index = l_columns.index(l_filter_attr)
    l_out_attrs_indices = []
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # find column indices of key attr, filter attr and output attrs in rtable
    r_columns = list(rtable.columns.values)
    r_key_attr_index = r_columns.index(r_key_attr)
    r_filter_attr_index = r_columns.index(r_filter_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    # convert ltable into a list of tuples
    ltable_list = convert_dataframe_to_list(ltable, l_filter_attr_index)

    # convert rtable into a list of tuples
    rtable_list = convert_dataframe_to_list(rtable, r_filter_attr_index)

    # generate token ordering using tokens in l_filter_attr and r_filter_attr
    token_ordering = gen_token_ordering_for_tables(
                                 [ltable_list, rtable_list],
                                 [l_filter_attr_index, r_filter_attr_index],
                                 position_filter.tokenizer,
                                 position_filter.sim_measure_type)

    handle_empty = (position_filter.allow_empty and
        position_filter.sim_measure_type not in ['OVERLAP', 'EDIT_DISTANCE'])

    # Build position index on l_filter_attr
    position_index = PositionIndex(ltable_list, l_filter_attr_index,
                                   position_filter.tokenizer,
                                   position_filter.sim_measure_type,
                                   position_filter.threshold, token_ordering)
    # While building the index, we cache the record ids with empty set of 
    # tokens. This is needed to handle the allow_empty flag.
    cached_data = position_index.build(handle_empty)
    l_empty_records = cached_data['empty_records']

    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)

    if show_progress:
        prog_bar = pyprind.ProgBar(len(rtable_list))

    for r_row in rtable_list:
        r_string = r_row[r_filter_attr_index]

        r_filter_attr_tokens = position_filter.tokenizer.tokenize(r_string)
        r_ordered_tokens = order_using_token_ordering(r_filter_attr_tokens,
                                                      token_ordering)

        if handle_empty and len(r_ordered_tokens) == 0:
            for l_id in l_empty_records:
                if has_output_attributes:
                    output_row = get_output_row_from_tables(
                                     ltable_list[l_id], r_row,
                                     l_key_attr_index, r_key_attr_index,
                                     l_out_attrs_indices,
                                     r_out_attrs_indices)
                else:
                    output_row = [ltable_list[l_id][l_key_attr_index],
                                  r_row[r_key_attr_index]]

                output_rows.append(output_row)
            continue

        candidate_overlap = position_filter.find_candidates(
                                r_ordered_tokens, position_index)

        for cand, overlap in iteritems(candidate_overlap):
            if overlap > 0:
                if has_output_attributes:
                    output_row = get_output_row_from_tables(
                                     ltable_list[cand], r_row,
                                     l_key_attr_index, r_key_attr_index, 
                                     l_out_attrs_indices, r_out_attrs_indices)
                else:
                    output_row = [ltable_list[cand][l_key_attr_index],
                                  r_row[r_key_attr_index]]

                output_rows.append(output_row)

        if show_progress:                    
            prog_bar.update()

    output_header = get_output_header_from_tables(l_key_attr, r_key_attr,
                                                  l_out_attrs, r_out_attrs, 
                                                  l_out_prefix, r_out_prefix)

    # generate a dataframe from the list of output rows
    output_table = pd.DataFrame(output_rows, columns=output_header)
    return output_table
