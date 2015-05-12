### Replace missing values/unknowns encoded as 88 with mean

import numpy as np
from sklearn.preprocessing import Imputer


def replace_missings(testing_data):
	testing_data[testing_data==88]=np.nan
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	imp.fit(testing_data)
	testing_data = imp.transform(testing_data)
	return testing_data

def replace_missings_thematic(testing_data):
	for i in range(len(testing_data)):
		testing_data[i] = replace_missings(testing_data[i])
	return testing_data	

