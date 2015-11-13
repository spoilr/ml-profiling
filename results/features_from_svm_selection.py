'''
Features after svm_feature_selection function
'''

# ####### net FEATURES ####### 15
# net_90 = set(['LettersPost', 'Education', 'InteractNet', 'RelatStat', 'ParRelatStat', 'Getaway', 'Involve', 'ClaimResp', 'NewMedia', 'OtherInv', 'DryRuns', 'TargetTyp', 'Interrupt', 'PersRelat', 'Stockpile'])
# ####### ill FEATURES ####### 22
# ill_90 = set(['Discriminate', 'Stress', 'Obsess', 'RelatStat', 'DryRuns', 'Insanity', 'ClaimResp', 'Interrupt', 'MentalIll', 'LettersPost', 'BombManuals', 'ParRelatStat', 'Getaway', 'Stockpile', 'SubstanceUse', 'Financial', 'SubAbuse', 'NewMedia', 'TargetTyp', 'AwareGriev', 'HarmVictimHelpless', 'Education'])
# ####### ideo FEATURES ####### 23
# ideo_90 = set(['Religion', 'AwareIdeo', 'OwnProp', 'ReligChangeInt', 'IdeoChangeInt', 'Interrupt', 'WarningLettersStatements', 'LettersPost', 'Contradict', 'ParRelatStat', 'Legitimise', 'Ideology', 'Stockpile', 'Implement', 'Financial', 'LocPubPriv', 'WideGroup', 'NewMedia', 'TargetTyp', 'DryRuns', 'Education', 'Children', 'BeliefChange'])



# ####### net FEATURES ####### 21
# net_70 = set(['LettersPost', 'Education', 'Financial', 'LiveAlone', 'BombManuals', 'InteractNet', 'RelatStat', 'ParRelatStat', 'Getaway', 'Virtual', 'Involve', 'ClaimResp', 'NewMedia', 'OtherInv', 'TargetTyp', 'DryRuns', 'MultiAttackMeth', 'RecruitNetGroup', 'Interrupt', 'PersRelat', 'Stockpile'])
# ####### ill FEATURES ####### 28
# ill_70 = set(['Discriminate', 'Stress', 'Virtual', 'MultiAttackMeth', 'Obsess', 'RelatStat', 'DryRuns', 'Insanity', 'ClaimResp', 'Interrupt', 'WarningLettersStatements', 'MentalIll', 'LettersPost', 'ParRelatStat', 'BombManuals', 'Violence', 'Getaway', 'Stockpile', 'Tipping', 'SubstanceUse', 'Financial', 'LiveAlone', 'SubAbuse', 'NewMedia', 'TargetTyp', 'AwareGriev', 'HarmVictimHelpless', 'Education'])
# ####### ideo FEATURES ####### 34
# ideo_70 = set(['FurtherAttacks', 'LettersPost', 'Virtual', 'Religion', 'AwareIdeo', 'PossessStories', 'MultiAttackMeth', 'RelatStat', 'OwnProp', 'NewMedia', 'IdeoChangeInt', 'ClaimResp', 'Interrupt', 'WarningLettersStatements', 'Training', 'BombManuals', 'Contradict', 'ParRelatStat', 'Legitimise', 'Getaway', 'Ideology', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Financial', 'LocPubPriv', 'CrimCon', 'WideGroup', 'ReligChangeInt', 'TargetTyp', 'DryRuns', 'Education', 'Children', 'BeliefChange'])



# ####### net FEATURES ####### 25
# net_50 = set(['InteractNet', 'Virtual', 'OtherInv', 'MultiAttackMeth', 'RelatStat', 'DryRuns', 'ClaimResp', 'Interrupt', 'PersRelat', 'LettersPost', 'BombManuals', 'OccCat', 'ParRelatStat', 'Getaway', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Financial', 'LiveAlone', 'CrimCon', 'OtherKnowledge', 'Involve', 'NewMedia', 'TargetTyp', 'Education'])
# ####### ill FEATURES ####### 33
# ill_50 = set(['Discriminate', 'Stress', 'Virtual', 'MultiAttackMeth', 'Obsess', 'RelatStat', 'DryRuns', 'Insanity', 'ClaimResp', 'Interrupt', 'WarningLettersStatements', 'MentalIll', 'LettersPost', 'ParRelatStat', 'BombManuals', 'OccCat', 'Violence', 'Getaway', 'Stockpile', 'Implement', 'Tipping', 'SubstanceUse', 'Financial', 'LiveAlone', 'CrimCon', 'OtherKnowledge', 'SubAbuse', 'NewMedia', 'TargetTyp', 'AwareGriev', 'HarmVictimHelpless', 'Education', 'Children'])
# ####### ideo FEATURES ####### 38
# ideo_50 = set(['FurtherAttacks', 'Training', 'Virtual', 'Religion', 'AwareIdeo', 'PossessStories', 'MultiAttackMeth', 'RelatStat', 'OwnProp', 'Denounce', 'IdeoChangeInt', 'ClaimResp', 'OccCat', 'Interrupt', 'WarningLettersStatements', 'LettersPost', 'BombManuals', 'Contradict', 'ParRelatStat', 'Legitimise', 'Getaway', 'Ideology', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Education', 'Financial', 'LocPubPriv', 'LiveAlone', 'CrimCon', 'WideGroup', 'OtherKnowledge', 'ReligChangeInt', 'TargetTyp', 'DryRuns', 'NewMedia', 'Children', 'BeliefChange'])



# ####### FEATURES ####### 43
# single_features_90 = set(['Discriminate', 'InteractNet', 'LettersPost', 'Religion', 'OtherInv', 'AwareIdeo', 'PossessStories', 'Insanity', 'Obsess', 'Involve', 'RelatStat', 'OwnProp', 'DryRuns', 'IdeoChangeInt', 'ClaimResp', 'Interrupt', 'PersRelat', 'WarningLettersStatements', 'MentalIll', 'Stress', 'BombManuals', 'Contradict', 'ParRelatStat', 'Legitimise', 'Getaway', 'ReligChangeInt', 'Ideology', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Tipping', 'SubstanceUse', 'Financial', 'LocPubPriv', 'WideGroup', 'SubAbuse', 'NewMedia', 'TargetTyp', 'AwareGriev', 'HarmVictimHelpless', 'Education', 'Children', 'BeliefChange'])

# ####### FEATURES ####### 53
# single_features_70 = set(['Discriminate', 'InteractNet', 'FurtherAttacks', 'Training', 'Virtual', 'Denounce', 'Religion', 'OtherInv', 'AwareIdeo', 'PossessStories', 'MultiAttackMeth', 'Insanity', 'Obsess', 'Involve', 'RelatStat', 'OwnProp', 'DryRuns', 'IdeoChangeInt', 'ClaimResp', 'HurtOthers', 'PersRelat', 'WarningLettersStatements', 'MentalIll', 'Stress', 'Violence', 'BombManuals', 'Contradict', 'ParRelatStat', 'Legitimise', 'Getaway', 'ReligChangeInt', 'Ideology', 'Interrupt', 'Stockpile', 'RecruitNetGroup', 'LettersPost', 'Implement', 'Tipping', 'SubstanceUse', 'Financial', 'LocPubPriv', 'LiveAlone', 'CrimCon', 'WideGroup', 'OtherKnowledge', 'SubAbuse', 'NewMedia', 'TargetTyp', 'AwareGriev', 'HarmVictimHelpless', 'Education', 'Children', 'BeliefChange'])

# ####### FEATURES ####### 57
# single_features_50 = set(['MilExp', 'Discriminate', 'InteractNet', 'FurtherAttacks', 'Isolated', 'Virtual', 'Denounce', 'LocPubPriv', 'Religion', 'OtherInv', 'AwareIdeo', 'PossessStories', 'MultiAttackMeth', 'Insanity', 'Obsess', 'Involve', 'RelatStat', 'OwnProp', 'DryRuns', 'IdeoChangeInt', 'ClaimResp', 'Contradict', 'Interrupt', 'PersRelat', 'WarningLettersStatements', 'MentalIll', 'Stress', 'Violence', 'Training', 'BombManuals', 'OccCat', 'ParRelatStat', 'Legitimise', 'Getaway', 'ReligChangeInt', 'Ideology', 'HurtOthers', 'Stockpile', 'RecruitNetGroup', 'LettersPost', 'Implement', 'Tipping', 'SubstanceUse', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'WideGroup', 'OtherKnowledge', 'SubAbuse', 'NewMedia', 'TargetTyp', 'AwareGriev', 'HarmVictimHelpless', 'Education', 'Children', 'BeliefChange'])


##### 6
net_90 = set(['InteractNet', 'RelatStat', 'DryRuns', 'Interrupt', 'Education', 'Children'])
##### 5
ill_90 = set(['MentalIll', 'DryRuns', 'Education', 'Children', 'RelatStat'])
##### 10
ideo_90 = set(['Regret', 'Contradict', 'Ideology', 'Religion', 'DryRuns', 'RelatStat', 'Interrupt', 'Education', 'Children', 'BeliefChange'])

##### 9
net_70 = set(['Regret', 'InteractNet', 'OccCat', 'CrimCon', 'DryRuns', 'RelatStat', 'Interrupt', 'Education', 'Children'])
##### 12
ill_70 = set(['MentalIll', 'Regret', 'LiveAlone', 'CrimCon', 'OccCat', 'DryRuns', 'AwareGriev', 'RelatStat', 'Interrupt', 'Education', 'Children', 'Obsess'])
##### 15
ideo_70 = set(['Regret', 'CrimCon', 'OccCat', 'RecruitNetGroup', 'RelatStat', 'Ideology', 'Religion', 'DryRuns', 'Adoption', 'Contradict', 'Interrupt', 'AwareIdeo', 'Education', 'Children', 'BeliefChange'])



##### 11
net_50 = set(['Regret', 'InteractNet', 'OccCat', 'RecruitNetGroup', 'CrimCon', 'DryRuns', 'RelatStat', 'Interrupt', 'LifeAspectChange', 'Education', 'Children'])
##### 13
ill_50 = set(['MentalIll', 'Regret', 'LiveAlone', 'CrimCon', 'OccCat', 'DryRuns', 'AwareGriev', 'RelatStat', 'Interrupt', 'LifeAspectChange', 'Education', 'Children', 'Obsess'])
##### 17
ideo_50 = set(['Contradict', 'Regret', 'CrimCon', 'OccCat', 'RecruitNetGroup', 'Legitimise', 'DryRuns', 'Ideology', 'Religion', 'LocationNature', 'Adoption', 'RelatStat', 'Interrupt', 'AwareIdeo', 'Education', 'Children', 'BeliefChange'])



##### 21
single_features_90 = set(['MentalIll', 'Education', 'Regret', 'HurtOthers', 'InteractNet', 'Contradict', 'RecruitNetGroup', 'Denounce', 'CrimCon', 'Insanity', 'Ideology', 'Religion', 'DryRuns', 'AwareIdeo', 'AwareGriev', 'RelatStat', 'Interrupt', 'OccCat', 'PersRelat', 'Children', 'BeliefChange'])
######################
##### 29
single_features_70 = set(['InteractNet', 'Training', 'Religion', 'Adoption', 'Obsess', 'RelatStat', 'Denounce', 'Insanity', 'Contradict', 'HurtOthers', 'LifeAspectChange', 'PersRelat', 'AwareGriev', 'MentalIll', 'Stress', 'OccCat', 'Legitimise', 'Ideology', 'Interrupt', 'RecruitNetGroup', 'Regret', 'LiveAlone', 'CrimCon', 'LocationNature', 'DryRuns', 'AwareIdeo', 'Education', 'Children', 'BeliefChange'])
######################
##### 33
single_features_50 = set(['InteractNet', 'Training', 'Religion', 'OtherInv', 'Adoption', 'Obsess', 'RelatStat', 'OwnProp', 'Denounce', 'Insanity', 'LocationNature', 'Contradict', 'Interrupt', 'LifeAspectChange', 'PersRelat', 'AwareGriev', 'MentalIll', 'Stress', 'OccCat', 'Legitimise', 'Ideology', 'HurtOthers', 'RecruitNetGroup', 'Tipping', 'Regret', 'LiveAlone', 'CrimCon', 'ReligChangeInt', 'DryRuns', 'AwareIdeo', 'Education', 'Children', 'BeliefChange'])