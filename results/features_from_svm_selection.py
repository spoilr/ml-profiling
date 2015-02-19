'''
Features after svm_feature_selection function
'''

####### net FEATURES ####### 29 
net_90 = set(['InteractNet', 'FurtherAttacks', 'LAKilled', 'OtherInv', 'MultiAttackMeth', 'RelatStat', 'DryRuns', 'ClaimResp', 'Interrupt', 'PersRelat', 'WarningLettersStatements', 'LettersPost', 'Violence', 'BombManuals', 'ParRelatStat', 'Getaway', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'NotCareInjustice', 'Involve', 'NewMedia', 'TargetTyp', 'Education', 'Children'])
####### ill FEATURES ####### 34 
ill_90 = set(['FurtherAttacks', 'Stress', 'LAKilled', 'MultiAttackMeth', 'Obsess', 'RelatStat', 'NewMedia', 'Insanity', 'ClaimResp', 'HurtOthers', 'WarningLettersStatements', 'MentalIll', 'LettersPost', 'Violence', 'BombManuals', 'ParRelatStat', 'Getaway', 'Isolated', 'Interrupt', 'Stockpile', 'Implement', 'Tipping', 'SubstanceUse', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'SubAbuse', 'DryRuns', 'TargetTyp', 'AwareGriev', 'HarmVictimHelpless', 'Education', 'Children'])
####### ideo FEATURES ####### 39 
ideo_90 = set(['FurtherAttacks', 'LettersPost', 'Virtual', 'Denounce', 'LocPubPriv', 'ReligChangeInt', 'LAKilled', 'AwareIdeo', 'PossessStories', 'MultiAttackMeth', 'RelatStat', 'OwnProp', 'NewMedia', 'IdeoChangeInt', 'ClaimResp', 'Interrupt', 'WarningLettersStatements', 'Training', 'ParRelatStat', 'BombManuals', 'Contradict', 'Violence', 'Legitimise', 'Getaway', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Propaganda', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'NotCareInjustice', 'DryRuns', 'TargetTyp', 'Adoption', 'Education', 'Children', 'BeliefChange'])



####### net FEATURES ####### 30 
net_70 = set(['InteractNet', 'FurtherAttacks', 'Virtual', 'LAKilled', 'OtherInv', 'MultiAttackMeth', 'RelatStat', 'DryRuns', 'ClaimResp', 'Interrupt', 'PersRelat', 'WarningLettersStatements', 'LettersPost', 'Violence', 'BombManuals', 'ParRelatStat', 'Getaway', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'NotCareInjustice', 'Involve', 'NewMedia', 'TargetTyp', 'Education', 'Children'])
####### ill FEATURES ####### 36 
ill_70 = set(['FurtherAttacks', 'Stress', 'Virtual', 'LAKilled', 'MultiAttackMeth', 'Obsess', 'RelatStat', 'NewMedia', 'Insanity', 'ClaimResp', 'HurtOthers', 'WarningLettersStatements', 'MentalIll', 'LettersPost', 'Violence', 'BombManuals', 'ParRelatStat', 'Getaway', 'Isolated', 'Interrupt', 'Stockpile', 'Implement', 'Tipping', 'SubstanceUse', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'NotCareInjustice', 'SubAbuse', 'DryRuns', 'TargetTyp', 'AwareGriev', 'HarmVictimHelpless', 'Education', 'Children'])
####### ideo FEATURES ####### 40 
ideo_70 = set(['FurtherAttacks', 'LettersPost', 'Virtual', 'Denounce', 'LocPubPriv', 'ReligChangeInt', 'LAKilled', 'AwareIdeo', 'PossessStories', 'MultiAttackMeth', 'RelatStat', 'OwnProp', 'NewMedia', 'IdeoChangeInt', 'ClaimResp', 'Interrupt', 'WarningLettersStatements', 'Training', 'ParRelatStat', 'BombManuals', 'Contradict', 'Violence', 'Legitimise', 'Getaway', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Propaganda', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'WideGroup', 'NotCareInjustice', 'DryRuns', 'TargetTyp', 'Adoption', 'Education', 'Children', 'BeliefChange'])



####### net FEATURES ####### 31 
net_50 = set(['InteractNet', 'OtherKnowledge', 'FurtherAttacks', 'Virtual', 'LAKilled', 'OtherInv', 'MultiAttackMeth', 'RelatStat', 'DryRuns', 'ClaimResp', 'Interrupt', 'PersRelat', 'WarningLettersStatements', 'LettersPost', 'Violence', 'BombManuals', 'ParRelatStat', 'Getaway', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'NotCareInjustice', 'Involve', 'NewMedia', 'TargetTyp', 'Education', 'Children'])
####### ill FEATURES ####### 36 
ill_50 = set(['FurtherAttacks', 'Stress', 'Virtual', 'LAKilled', 'MultiAttackMeth', 'Obsess', 'RelatStat', 'NewMedia', 'Insanity', 'ClaimResp', 'HurtOthers', 'WarningLettersStatements', 'MentalIll', 'LettersPost', 'Violence', 'BombManuals', 'ParRelatStat', 'Getaway', 'Isolated', 'Interrupt', 'Stockpile', 'Implement', 'Tipping', 'SubstanceUse', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'NotCareInjustice', 'SubAbuse', 'DryRuns', 'TargetTyp', 'AwareGriev', 'HarmVictimHelpless', 'Education', 'Children'])
####### ideo FEATURES ####### 43 
ideo_50 = set(['MilExp', 'NotCareInjustice', 'FurtherAttacks', 'LettersPost', 'Virtual', 'Denounce', 'LocPubPriv', 'ReligChangeInt', 'LAKilled', 'AwareIdeo', 'PossessStories', 'MultiAttackMeth', 'RelatStat', 'OwnProp', 'DryRuns', 'IdeoChangeInt', 'ClaimResp', 'Interrupt', 'WarningLettersStatements', 'Training', 'ParRelatStat', 'BombManuals', 'Contradict', 'Violence', 'Legitimise', 'Getaway', 'Ideology', 'Stockpile', 'RecruitNetGroup', 'Implement', 'Propaganda', 'Financial', 'Regret', 'LiveAlone', 'CrimCon', 'WideGroup', 'OtherKnowledge', 'NewMedia', 'TargetTyp', 'Adoption', 'Education', 'Children', 'BeliefChange'])

####### FEATURES ####### 51
 single_features = set(['InteractNet', 'FurtherAttacks', 'Isolated', 'ReligChangeInt', 'LAKilled', 'OtherInv', 'AwareIdeo', 'PossessStories', 'MultiAttackMeth', 'Insanity', 'Obsess', 'SubAbuse', 'LiveAlone', 'RelatStat', 'OwnProp', 'DryRuns', 'IdeoChangeInt', 'ClaimResp', 'HurtOthers', 'PersRelat', 'WarningLettersStatements', 'MentalIll', 'Stress', 'ParRelatStat', 'Training', 'BombManuals', 'Contradict', 'Violence', 'Legitimise', 'Getaway', 'Interrupt', 'Stockpile', 'RecruitNetGroup', 'LettersPost', 'Implement', 'Tipping', 'Propaganda', 'SubstanceUse', 'Financial', 'Regret', 'LocPubPriv', 'CrimCon', 'Involve', 'NewMedia', 'TargetTyp', 'AwareGriev', 'Adoption', 'HarmVictimHelpless', 'Education', 'Children', 'BeliefChange'])