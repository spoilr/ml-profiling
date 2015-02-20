mapping = dict()

mapping['RelatStat']='RelS'
mapping['Children']='Ch'
mapping['ParRelatStat']='ParS'
mapping['Education']='Ed'
mapping['OccCat']='Occ'
mapping['MilExp']='Mil'
mapping['CrimCon']='Crim'
mapping['Ideology']='Ideo'
mapping['Religion']='Rel'
mapping['WarningLettersStatements']='Warn'
mapping['AwareGriev']='AGr'
mapping['AwareIdeo']='AId'
mapping['IdeoChangeInt']='IdeoCI'
mapping['ReligChangeInt']='RelCI'
mapping['LifeAspectChange']='Life'
mapping['Legitimise']='Legit'
mapping['Funds']='Fund'
mapping['Denounce']='Den'
mapping['LiveAlone']='Alone'
mapping['Adoption']='Adopt'
mapping['Training']='Train'
mapping['Virtual']='Virt'
mapping['SubAbuse']='Subs'
mapping['MentalIll']='Mental'
mapping['Isolated']='Iso'
mapping['DryRuns']='Dry'
mapping['BombManuals']='Bomb'
mapping['WideGroup']='WideGr'
mapping['NewMedia']='Media'
mapping['Tipping']='Tip'
mapping['Interrupt']='Inter'
mapping['NotCareInjustice']='Injust'
mapping['HarmVictimHelpless']='Harm'
mapping['PersRelat']='PersS'
mapping['Financial']='Fin'
mapping['HurtOthers']='Hurt'
mapping['Stress']='Stress'
mapping['SubstanceUse']='SubUse'
mapping['Violence']='Violence'
mapping['TargetTyp']='TarType'
mapping['LocationNature']='LocNat'
mapping['LocPubPriv']='PubPriv'
mapping['Stockpile']='Stock'
mapping['Contradict']='Contr'
mapping['Obsess']='Obs'
mapping['Regret']='Reg'
mapping['BeliefChange']='Bel'
mapping['Insanity']='Ins'
mapping['Implement']='Impl'
mapping['MultiAttackMeth']='MultiAtt'
mapping['Discriminate']='Discr'
mapping['LettersPost']='Let'
mapping['Getaway']='Get'
mapping['MultiEventTarget']='MultiEv'
mapping['Involve']='Inv'
mapping['InteractNet']='InterNet'
mapping['OtherInv']='OtherInv'
mapping['OtherKnowledge']='OtherKnow'
mapping['RecruitNetGroup']='RecruitNet'
mapping['Propaganda']='Prop'
mapping['OwnProp']='OwnProp'
mapping['LAKilled']='LAKill'
mapping['FurtherAttacks']='FurtherAtt'
mapping['ClaimResp']='Claim'
mapping['PossessStories']='Stories'
mapping['HighValueCivilian']='ValCivil'

inv_mapping = {v: k for k, v in mapping.items()}

binary_convertor = dict()
binary_convertor['MultiEv'] = ['SingleEv', 'MultiEvNoTarCh', 'MultiEvTarCh']
binary_convertor['LocNat'] = ['GovLocNat', 'BusLocNat', 'CitLocNat', 'RelLocNat', 'MilLocNat', 'OtherLocNat']
binary_convertor['TarType'] = ['PplTarTyp', 'PropTarTyp', 'BothTarTyp']
binary_convertor['Fund'] = ['NoFin', 'IllFin', 'LegFin', 'BothFin']
binary_convertor['Life'] = ['LifeNo', 'LifeWS', 'LifeFam', 'LifeFin', 'LifeUpEv', 'LifeWSFin', 'LifeWSFam', 'LifeWSUpEv', 'LifeWSFinUpEv']
binary_convertor['RelCI'] = ['NoChRel', 'NoIntRel', 'YesChRel', 'YesIntRel']
binary_convertor['IdeoCI'] = ['NoChIdeo', 'NoIntIdeo', 'YesChIdeo', 'YesIntIdeo']
binary_convertor['Rel'] = ['UnkRel', 'Muslim', 'Jewish', 'Atheist', 'OtherRel']
binary_convertor['Ideo'] = ['RightW', 'SingleIs', 'Religious']
binary_convertor['Crim'] = ['NoCrimConv', 'YesNoImp', 'YesImp']
binary_convertor['Mil'] = ['NoMil', 'NoMilRej']
binary_convertor['Occ'] = ['Unemp', 'Student', 'ServInd', 'Prof', 'Constr', 'ClSAd', 'Agric', 'Other']
binary_convertor['Ed'] = ['NoHS', 'SomeHS', 'GradHS', 'GED', 'NoDeg', 'Deg', 'NoCol', 'Bac', 'GradNoDeg', 'Msc', 'Phd']
binary_convertor['ParS'] = ['PNMarried', 'PMarried', 'PSeparated', 'PDiv', 'PWid', 'POther']
binary_convertor['RelS'] = ['Single', 'Partner', 'Married', 'Separated', 'Divorced']




inv_binary_convertor = dict()

def reverse_binary_convertor():
	for k,v in binary_convertor.items():
		for x in v:
			inv_binary_convertor[x] = k

reverse_binary_convertor()		
