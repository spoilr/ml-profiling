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


##### 19
net_90 = set(['LettersPost', 'Education', 'Discriminate', 'LiveAlone', 'BombManuals', 'InteractNet', 'OtherKnowledge', 'FurtherAttacks', 'NotCareInjustice', 'Getaway', 'Involve', 'NewMedia', 'Stockpile', 'DryRuns', 'RecruitNetGroup', 'Interrupt', 'Financial', 'TargetTyp', 'WarningLettersStatements'])
##### 20
ill_90 = set(['MentalIll', 'Stress', 'Education', 'Financial', 'LiveAlone', 'BombManuals', 'LettersPost', 'FurtherAttacks', 'NotCareInjustice', 'Isolated', 'Getaway', 'DryRuns', 'HurtOthers', 'NewMedia', 'Stockpile', 'OtherKnowledge', 'Interrupt', 'TargetTyp', 'WarningLettersStatements', 'AwareGriev'])
##### 25
ideo_90 = set(['Discriminate', 'NotCareInjustice', 'FurtherAttacks', 'Religion', 'AwareIdeo', 'RelatStat', 'Denounce', 'IdeoChangeInt', 'Interrupt', 'LettersPost', 'BombManuals', 'Contradict', 'Legitimise', 'Getaway', 'Ideology', 'Stockpile', 'RecruitNetGroup', 'Propaganda', 'Financial', 'LiveAlone', 'LocPubPriv', 'OtherKnowledge', 'ReligChangeInt', 'TargetTyp', 'Education'])

##### 24
net_70 = set(['Discriminate', 'InteractNet', 'NotCareInjustice', 'FurtherAttacks', 'MultiEventTarget', 'RelatStat', 'DryRuns', 'Interrupt', 'WarningLettersStatements', 'LettersPost', 'BombManuals', 'ParRelatStat', 'Getaway', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Financial', 'LiveAlone', 'OtherKnowledge', 'Involve', 'NewMedia', 'TargetTyp', 'Education', 'Children'])
##### 30
ill_70 = set(['Discriminate', 'OtherKnowledge', 'FurtherAttacks', 'LettersPost', 'MultiAttackMeth', 'MultiEventTarget', 'DryRuns', 'Insanity', 'HurtOthers', 'LifeAspectChange', 'WarningLettersStatements', 'MentalIll', 'Stress', 'Violence', 'BombManuals', 'ParRelatStat', 'Getaway', 'Isolated', 'Interrupt', 'Stockpile', 'Implement', 'SubstanceUse', 'Financial', 'LiveAlone', 'NotCareInjustice', 'NewMedia', 'TargetTyp', 'AwareGriev', 'Education', 'Children'])
##### 34
ideo_70 = set(['MilExp', 'Discriminate', 'NotCareInjustice', 'FurtherAttacks', 'Religion', 'AwareIdeo', 'MultiEventTarget', 'RelatStat', 'Denounce', 'IdeoChangeInt', 'Interrupt', 'LifeAspectChange', 'WarningLettersStatements', 'LettersPost', 'BombManuals', 'Contradict', 'ParRelatStat', 'Legitimise', 'Getaway', 'ReligChangeInt', 'Ideology', 'Stockpile', 'RecruitNetGroup', 'Propaganda', 'Financial', 'LiveAlone', 'LocPubPriv', 'OtherKnowledge', 'NewMedia', 'TargetTyp', 'DryRuns', 'Education', 'Children', 'BeliefChange'])



##### 28
net_50 = set(['Discriminate', 'InteractNet', 'OtherKnowledge', 'FurtherAttacks', 'MultiAttackMeth', 'MultiEventTarget', 'RelatStat', 'DryRuns', 'Interrupt', 'LifeAspectChange', 'WarningLettersStatements', 'LettersPost', 'Violence', 'BombManuals', 'ParRelatStat', 'Getaway', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Financial', 'LiveAlone', 'CrimCon', 'NotCareInjustice', 'Involve', 'NewMedia', 'TargetTyp', 'Education', 'Children'])
##### 33
ill_50 = set(['Discriminate', 'OtherKnowledge', 'FurtherAttacks', 'LettersPost', 'MultiAttackMeth', 'Obsess', 'MultiEventTarget', 'RelatStat', 'DryRuns', 'Insanity', 'HurtOthers', 'LifeAspectChange', 'WarningLettersStatements', 'MentalIll', 'Stress', 'Violence', 'BombManuals', 'ParRelatStat', 'Getaway', 'Isolated', 'Interrupt', 'Stockpile', 'Implement', 'Tipping', 'SubstanceUse', 'Financial', 'LiveAlone', 'NotCareInjustice', 'NewMedia', 'TargetTyp', 'AwareGriev', 'Education', 'Children'])
##### 39
ideo_50 = set(['MilExp', 'Discriminate', 'NotCareInjustice', 'FurtherAttacks', 'LettersPost', 'Religion', 'AwareIdeo', 'MultiAttackMeth', 'MultiEventTarget', 'RelatStat', 'Denounce', 'IdeoChangeInt', 'Interrupt', 'LifeAspectChange', 'WarningLettersStatements', 'Training', 'Violence', 'BombManuals', 'Contradict', 'ParRelatStat', 'Legitimise', 'Getaway', 'ReligChangeInt', 'Ideology', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Propaganda', 'Financial', 'LiveAlone', 'LocPubPriv', 'WideGroup', 'OtherKnowledge', 'NewMedia', 'TargetTyp', 'DryRuns', 'Education', 'Children', 'BeliefChange'])



##### 40
single_features_90 = set(['MilExp', 'Discriminate', 'InteractNet', 'NotCareInjustice', 'FurtherAttacks', 'Isolated', 'Religion', 'OtherInv', 'AwareIdeo', 'MultiAttackMeth', 'Insanity', 'RelatStat', 'Denounce', 'IdeoChangeInt', 'Interrupt', 'WarningLettersStatements', 'MentalIll', 'Stress', 'BombManuals', 'Contradict', 'Legitimise', 'Getaway', 'ReligChangeInt', 'Ideology', 'HurtOthers', 'Stockpile', 'RecruitNetGroup', 'LettersPost', 'Implement', 'Propaganda', 'NewMedia', 'Financial', 'LocPubPriv', 'LiveAlone', 'OtherKnowledge', 'Involve', 'DryRuns', 'TargetTyp', 'AwareGriev', 'Education'])
######################
##### 50
single_features_70 = set(['MilExp', 'Discriminate', 'InteractNet', 'NotCareInjustice', 'FurtherAttacks', 'Isolated', 'Religion', 'OtherInv', 'AwareIdeo', 'MultiAttackMeth', 'Insanity', 'Obsess', 'MultiEventTarget', 'RelatStat', 'Denounce', 'IdeoChangeInt', 'Interrupt', 'LifeAspectChange', 'WarningLettersStatements', 'MentalIll', 'Stress', 'Violence', 'Training', 'BombManuals', 'Contradict', 'ParRelatStat', 'Legitimise', 'Getaway', 'ReligChangeInt', 'Ideology', 'HurtOthers', 'Stockpile', 'RecruitNetGroup', 'LettersPost', 'Implement', 'Propaganda', 'SubstanceUse', 'NewMedia', 'Financial', 'LocPubPriv', 'LiveAlone', 'WideGroup', 'OtherKnowledge', 'Involve', 'DryRuns', 'TargetTyp', 'AwareGriev', 'Education', 'Children', 'BeliefChange'])
######################
##### 53
single_features_50 = set(['MilExp', 'Discriminate', 'InteractNet', 'NotCareInjustice', 'FurtherAttacks', 'Isolated', 'Religion', 'OtherInv', 'AwareIdeo', 'MultiAttackMeth', 'Insanity', 'Obsess', 'MultiEventTarget', 'RelatStat', 'Denounce', 'IdeoChangeInt', 'Interrupt', 'LifeAspectChange', 'PersRelat', 'WarningLettersStatements', 'MentalIll', 'Stress', 'Violence', 'Training', 'BombManuals', 'Contradict', 'ParRelatStat', 'Legitimise', 'Getaway', 'ReligChangeInt', 'Ideology', 'HurtOthers', 'Stockpile', 'RecruitNetGroup', 'LettersPost', 'Implement', 'Tipping', 'Propaganda', 'SubstanceUse', 'NewMedia', 'Financial', 'LocPubPriv', 'LiveAlone', 'CrimCon', 'WideGroup', 'OtherKnowledge', 'Involve', 'DryRuns', 'TargetTyp', 'AwareGriev', 'Education', 'Children', 'BeliefChange'])