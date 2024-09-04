spert_configs={
    "SPACY_MODEL" : "en_core_web_sm",
    "TOKEN":"tokens",
    "ENTITY":"entities",
    "RELATION":"relations",
    "ID":"orig_id",
    "SENTENCE_ID":"sent_id",
    "MAXLEN":1000,
    "SENTENCE_SEP":[".",","]
}

ASSERTION="Assertion"
PROBLEM="Problem"
DRUG='Drug'
DRUG_CANDIDATES=['Possible_Drug'] # to rename those candidates as drug

ANATOMY="Anatomy"
FREQUENCY="Frequency"
DURATION="Duration"
CHARACTERISTICS="Characteristics"
CHANGE="Change"
SEVERITY="Severity"

all_attributes=[ASSERTION,ANATOMY,DURATION,FREQUENCY,CHARACTERISTICS,CHANGE,SEVERITY]
Sub_types={
    ASSERTION: ['present','absent','hypothetical','not_patient','possible','conditional'],
    CHANGE: ['improving','worsening','no_change','resolved'],
    SEVERITY: ['mild','moderate','severe'],
}

SPAN_WITH_ASSERTION=[ASSERTION,CHANGE, SEVERITY]
VALID_TYPES=[PROBLEM,DRUG,
            ASSERTION,CHANGE,SEVERITY,
            ANATOMY,FREQUENCY,DURATION,CHARACTERISTICS]
TRIGGERS=[DRUG,PROBLEM]
relation_type_map={
    "admin_for":"TrAP",
    "not_admin_because":"TrNAP",
    "worsens":"TrWP",
    "worsens_or_NotImproving":"TrWP",
    "causes":"TrCP",
    "improves":"TrIP",
    "improves_or_NotWorsening":"TrIP",
    "PIP":"PIP",
    "TrAP":"TrAP",
    "TrNAP":"TrNAP",
    "TrWP":"TrWP",
    "TrCP":"TrCP",
    "TrIP":"TrIP"
}
entity_type_map={
    "Drug":"treatment",
    "Problem":"problem"
}

entity_type_map2={
    "Drug":"Drug",
    "Characteristics":"Characteristics",
    "Change":"Change",
    "Duration":"Duration",
    "Problem":"Problem",
    "Problem-present":"Problem-present",
    "Problem-absent":"Problem-absent",  
    "Problem-conditional":"Problem-conditional",
    "Problem-hypothetical":"Problem-hypothetical",
    "Problem-not_patient":"Problem-not_patient",
    "Problem-possible":"Problem-possible",
    "Anatomy":"Anatomy",
    "Severity":"Severity",
    "Frequency":"Frequency"
}
for v in ["Severity-mild" ,"Severity-moderate" ,"Severity-severe","Change-improving" ,"Change-no_change" ,"Change-resolved" ,"Change-worsening"]:
    entity_type_map2[v]=v

relation_type_map2={
    "Problem-Characteristics":"Problem-Characteristics",
    "Problem-Change":"Problem-Change",
    "Problem-Duration":"Problem-Duration",
    #"Problem-Assertion":"Problem-Assertion",
    "Problem-Anatomy":"Problem-Anatomy",
    "Problem-Severity":"Problem-Severity",
    "Problem-Frequency":"Problem-Frequency",
    "admin_for":"TrAP",
    "not_admin_because":"TrNAP",
    "worsens":"TrWP",
    "worsens_or_NotImproving":"TrWP",
    "causes":"TrCP",
    "improves":"TrIP",
    "improves_or_NotWorsening":"TrIP",
    "PIP":"PIP",
    "TrAP":"TrAP",
    "TrNAP":"TrNAP",
    "TrWP":"TrWP",
    "TrCP":"TrCP",
    "TrIP":"TrIP"
}

relation_names={
    'TrAP':'AdminFor',
    'TrNAP':'NotAdminBecause',
    'TrCP':'Causes',
    'TrIP':'Improves',
    'TrWP':'Worsens',
    'PIP':'PIP'
}

relation_names_reversed={}
for k, v in relation_names.items():
    relation_names_reversed[v]=k

Entity_types=['Assertion','Change','Severity']
Entity_subtypes=[]
for ent in entity_type_map2:
    if '-' in ent:
        t,st=ent.split('-')
        if t not in Entity_types:
            Entity_types.append(t)
        if st not in Entity_subtypes:
            Entity_subtypes.append(st)
    else:
        if ent not in Entity_types:
            Entity_types.append(ent)

choice_types=["A","B","C","D","E","F","G","H"]
relation_types=["TrAP",
           "TrNAP",
           "TrWP",
           "TrCP",
           "TrIP",
           "PIP"
           ]