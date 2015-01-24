def add_misclassified_ids(model, test_index, known_dataset, known_targets, ids):
	misclassified_ids = []
	for ind in test_index:
		pred = model.predict(known_dataset[ind])
		if pred != known_targets[ind]:
			misclassified_ids.append(ids[ind])	
	return misclassified_ids