def remove_exclude_attr(feature_attrs, exclude_attrs, dataset):
	for attr in exclude_attrs:                                          
    	feature_attrs.remove(attr)
    return feature_attrs 