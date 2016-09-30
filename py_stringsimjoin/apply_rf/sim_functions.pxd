
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef double jaccard(const vector[int]& tokens1, const vector[int]& tokens2) nogil

cdef double dice(const vector[int]& tokens1, const vector[int]& tokens2) nogil

cdef double cosine(const vector[int]& tokens1, const vector[int]& tokens2) nogil

cdef double edit_distance(const string& str1, const string& str2) nogil
