Gender = {'male':1, 'female':2}
inv_Gender = {v: k for k, v in Gender.items()}

POBCat = {'usa':1, 'non-usa':2}
inv_POBCat = {v: k for k, v in POBCat.items()}

Citizenship = {'us':1, 'other':2}
inv_Citizenship = {v: k for k, v in Citizenship.items()}

RelatStat = {'single':0, 'partnered':1, 'married':2, 'separated':3, 'divorced':4, 'widowed':5}
inv_RelatStat = {v: k for k, v in RelatStat.items()}

ParRelatStat = {'never_married':0, 'married':1, 'separated':2, 'divorced':3, 'widowed':4, 'other':5}
inv_ParRelatStat = {v: k for k, v in ParRelatStat.items()}

Education = {'no_high_school':0, 'some_high_school':1, 'graduated_high_school':2, 'ged':3, 'community_or_trade_college_no_degree': 4, 'community_or_trade_college_degree':5, '4y_college_no_degree':6, 'undegrad_or_bac_degree':7, 'graduate_school_no_degree':8, 'masters':9, 'phd':10, 'none':99}
inv_Education = {v: k for k, v in Education.items()}

OccCat = {'unemployed':0, 'student':1, 'service_industry':2, 'professional':3, 'construction':4, 'clerical/sales/admin':5, 'agriculture':6, 'other':7}
inv_OccCat = {v: k for k, v in OccCat.items()}

MilExp = {'no':1, 'no_rejected':2, 'yes':3}
inv_MilExp = {v: k for k, v in MilExp.items()}

CrimCon = {'no':1, 'yes_not_imprisoned':2, 'yes_imprisoned':3}
inv_CrimCon = {v: k for k, v in CrimCon.items()}

Ideology = {'nationalist':0, 'left_wing':1, 'right_wing':2, 'single_issue':3, 'religious':4, 'other':5}
inv_Ideology = {v: k for k, v in Ideology.items()}

Religion = {'christian':0, 'muslim':1, 'hindu':2, 'buddhist':3, 'sikhs':4, 'jewish':5, 'agnostic':6, 'atheist':7, 'other':8}
inv_Religion = {v: k for k, v in Religion.items()}

IdeoChangeInt = {'no':1, 'no_intensified':2, 'yes':3, 'yes_intensified':4}
inv_IdeoChangeInt = {v: k for k, v in IdeoChangeInt.items()}

ReligChangeInt = {'no':1, 'no_intensified':2, 'yes':3, 'yes_intensified':4}
inv_ReligChangeInt = {v: k for k, v in ReligChangeInt.items()}

LifeAspectChange = {'no_change': 0, 'work_school':1, 'family':2, 'financial':3, 'upcoming_event':4, 'work_school_financial':5, 'work_school_family':6, 'work_school_upcoming':7, 'work_school_financial_upcoming':8}
inv_LifeAspectChange = {v: k for k, v in LifeAspectChange.items()}

Funds = {'no':1, 'illegal':2, 'legal':3, 'both':4}
inv_Funds = {v: k for k, v in Funds.items()}

TargetTyp = {'people':0, 'property':1, 'both':2}
inv_TargetTyp = {v: k for k, v in TargetTyp.items()}

LocationNature = {'government':0, 'business':1, 'citizens':2, 'religious':3, 'military':4, 'other':5}
inv_LocationNature = {v: k for k, v in LocationNature.items()}

LocPubPriv = {'public':0, 'private':1}
inv_LocPubPriv = {v: k for k, v in LocPubPriv.items()}

TargetGroup = {'government':0, 'business':1, 'citizens':2, 'other':3}
inv_TargetGroup = {v: k for k, v in TargetGroup.items()}

Getaway = {'planned':0, 'opportunistic':1, 'not_applicable':89}
inv_Getaway = {v: k for k, v in Getaway.items()}

MultiEventTarget = {'single_event':1, 'multi_event_no_target_change':2, 'multi_event_target_change':3}
inv_MultiEventTarget = {v: k for k, v in MultiEventTarget.items()}

HighValueCivilian = {'highval':1, 'civil':2}
inv_HighValueCivilian = {v: k for k, v in HighValueCivilian.items()}

inv_explanatory_data = {'RelatStat':inv_RelatStat, 'ParRelatStat':inv_ParRelatStat, 'Education':inv_Education, 'OccCat':inv_OccCat, 
					'MilExp':inv_MilExp, 'CrimCon':inv_CrimCon, 'Ideology':inv_Ideology, 'Religion':inv_Religion, 
					'IdeoChangeInt':inv_IdeoChangeInt, 'ReligChangeInt':inv_ReligChangeInt, 'LifeAspectChange':inv_LifeAspectChange,
					'Funds':inv_Funds, 'TargetTyp':inv_TargetTyp, 'LocationNature':inv_LocationNature, 'LocPubPriv':inv_LocPubPriv,
					'Getaway':inv_Getaway, 'MultiEventTarget':inv_MultiEventTarget, 'HighValueCivilian':inv_HighValueCivilian}

explanatory_data = {'RelatStat':RelatStat, 'ParRelatStat':ParRelatStat, 'Education':Education, 'OccCat':OccCat, 
					'MilExp':MilExp, 'CrimCon':CrimCon, 'Ideology':Ideology, 'Religion':Religion, 
					'IdeoChangeInt':IdeoChangeInt, 'ReligChangeInt':ReligChangeInt, 'LifeAspectChange':LifeAspectChange,
					'Funds':Funds, 'TargetTyp':TargetTyp, 'LocationNature':LocationNature, 'LocPubPriv':LocPubPriv,
					'Getaway':Getaway, 'MultiEventTarget':MultiEventTarget, 'HighValueCivilian':HighValueCivilian}


