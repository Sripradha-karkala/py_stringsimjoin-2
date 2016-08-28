
enum similarity_measure_type {
  COSINE,
  DICE,
  EDIT_DISTANCE,
  JACCARD,
  OVERLAP,
  OVERLAP_COEFFICIENT
};

enum comparison_operator {
  EQ,
  GE,
  GT,
  LE,
  LT,
  NE
};

class Predicate {
  public:
    similarity_measure_type sim_type;
    comparison_operator comp_op;
    double threshold;
    
    Predicate(similarity_measure_type sim_type, comparison_operator comp_op, double threshold);
    ~Predicate();
};
