def save_output(file_name, f1_scores, error_rates, hp_rates, hr_rates, hf_rates, cp_rates, cr_rates, cf_rates, folds):
	with open(file_name, "a") as myfile:	
		myfile.write('\n##############################\n')
	with open(file_name, "a") as myfile:	
		myfile.write('\nFinal error %f' % (float(error_rates) / folds))
	with open(file_name, "a") as myfile:	
		myfile.write('\nFinal accuracy %f' % (1 - (float(error_rates) / folds)))	
	with open(file_name, "a") as myfile:	
		myfile.write('\nHighval precision %f' % (float(hp_rates) / folds))	
	with open(file_name, "a") as myfile:	
		myfile.write('\nHighval recall %f' % (float(hr_rates) / folds))	
	with open(file_name, "a") as myfile:	
		myfile.write('\nHighval f1 %f' % (float(hf_rates) / folds))	
	with open(file_name, "a") as myfile:	
		myfile.write('\nCivil precision %f' % (float(cp_rates) / folds))	
	with open(file_name, "a") as myfile:	
		myfile.write('\nCivil recall %f' % (float(cr_rates) / folds))	
	with open(file_name, "a") as myfile:	
		myfile.write('\nCivil f1 %f' % (float(cf_rates) / folds))