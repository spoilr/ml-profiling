# violence_theme = {'net': SVC(class_weight='auto', C=8.4, gamma=0.0084), 'ill': SVC(class_weight='auto', gamma=0.03), 'ideo': SVC(class_weight='auto', C=8.4, gamma=0.007)}
violence_theme = {'net': SVC(class_weight='auto', C=1.3, gamma=0.23), 'ill': SVC(class_weight='auto', C=0.3, gamma=0.009), 'ideo': SVC(class_weight='auto', C=2.4, gamma=0.007)}

# violence_theme_ft = {'net': SVC(class_weight='auto', C=1.7, gamma=0.01), 'ill': SVC(class_weight='auto', C=0.3, gamma=0.09), 'ideo': SVC(class_weight='auto', C=0.7, gamma=0.01)} # weights = [1,1,1]
violence_theme_ft = {'net': SVC(class_weight='auto', C=1.4, gamma=1.2), 'ill': SVC(class_weight='auto', C=0.6, gamma=0.01), 'ideo': SVC(class_weight='auto', C=0.8, gamma=0.01)} # weights = [1,1,1]

violence_single = {SVC(class_weight='auto', C=1.7, gamma=0.05)}

violence_single_ft = {KNeighborsClassifier(weights='distance', n_neighbors=7)}



discriminate_theme = {'net': SVC(class_weight='auto', C=1.2, gamma=0.0025), 'ill': SVC(class_weight='auto', C=0.8, gamma=0.063), 'ideo': SVC(class_weight='auto', C=1.3, gamma=0.001)}

discriminate_theme_ft = {'net': SVC(class_weight='auto', C=0.3, gamma=0.01), 'ill': SVC(class_weight='auto', C=0.5, gamma=0.03), 'ideo': SVC(class_weight='auto', C=0.5, gamma=0.003)}

discriminate_single = {SVC(class_weight='auto', C=0.7)}

discriminate_single_ft = {SVC(class_weight='auto', C=0.7, gamma=0.01)}

discriminate_theme = {DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)}