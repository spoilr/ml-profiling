violence_theme = {'net': SVC(class_weight='auto', C=8.4, gamma=0.0084), 'ill': SVC(class_weight='auto', gamma=0.03), 'ideo': SVC(class_weight='auto', C=8.4, gamma=0.007)}

violence_theme_ft = {'net': SVC(class_weight='auto', C=1.7, gamma=0.01), 'ill': SVC(class_weight='auto', C=0.3, gamma=0.09), 'ideo': SVC(class_weight='auto', C=0.7, gamma=0.01)} # weights = [1,1,1]

discriminate_theme = {'net': SVC(class_weight='auto', C=1.2, gamma=0.0025), 'ill': SVC(class_weight='auto', C=0.8, gamma=0.063), 'ideo': SVC(class_weight='auto', C=1.3, gamma=0.001)}

discriminate_theme_ft = {'net': SVC(class_weight='auto', C=0.3, gamma=0.01), 'ill': SVC(class_weight='auto', C=0.5, gamma=0.03), 'ideo': SVC(class_weight='auto', C=0.5, gamma=0.003)}