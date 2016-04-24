from py_stringsimjoin.filter.filter_utils import get_prefix_length
from py_stringsimjoin.index.index import Index
from py_stringsimjoin.utils.token_ordering import order_using_token_ordering


class PrefixIndex(Index):

    def __init__(self, table, id_attr, index_attr, tokenizer, 
                 sim_measure_type, threshold, token_ordering):
        self.table = table
        self.id_attr = id_attr
        self.index_attr = index_attr
        self.tokenizer = tokenizer
        self.sim_measure_type = sim_measure_type
        self.threshold = threshold
        self.token_ordering = token_ordering
        self.index = {}
        super(self.__class__, self).__init__()

    def build(self):
        for row in self.table:
            index_attr_tokens = order_using_token_ordering(set(
                                    self.tokenizer(str(row[self.index_attr]))), 
                                                           self.token_ordering)
            prefix_length = get_prefix_length(
                                len(index_attr_tokens),
                                self.sim_measure_type, self.threshold)
 
            row_id = row[self.id_attr]
            for token in index_attr_tokens[0:prefix_length]:
                if self.index.get(token) is None:
                    self.index[token] = []
                self.index.get(token).append(row_id)

        return True

    def probe(self, token):
        return self.index.get(token, [])