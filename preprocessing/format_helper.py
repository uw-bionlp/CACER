import re
from glob import glob
import random
import json
import numpy as np
import json
from tqdm import tqdm
import copy
import spacy
import os
import shutil
nlp = spacy.load("en_core_web_sm")

from config import *
def generate_options(span1,type1,span2,type2,rel,ent1,ent2,PIP_index=0):
    
    if type1==PROBLEM and type2==DRUG:
        span1,type1,span2,type2 = span2,type2,span1,type1
        ent1,ent2=ent2,ent1
    if rel is not None and rel not in relation_type_map:
        return span1,type1,span2,type2,None,None, ent1,ent2
    options=[]
    if type1==DRUG and type2==PROBLEM:
        options+=[f"{span1} is given as a treatment for {span2},  but the outcome is not mentioned in the sentence."]
        options+=[f"{span1} is not given or discontinued because of the {span2} that it does not cause."]
        options+=[f"{span1} is given as a treatment for {span2},  but {span1} does not cure the {span2}, does not improve the {span2}, or makes the {span2} worse."]
        options+=[f"{span1} is not given as a treatment for {span2}, but it causes {span2}."]
        options+=[f"{span1} improves, cures, stablize {span2}."]
    elif type1==PROBLEM and type2==PROBLEM:
        options+=[f"{span1} causes, describes or reveals aspects of {span2}."]
        options+=[f"{span2} causes, describes or reveals aspects of {span1}."]
    else:
        return span1,type1,span2,type2,None,None, ent1,ent2
    options+=[f"None of the above."]
    options=[f"({choice_types[idx]}) {O}" for idx,O in enumerate(options)]
    if rel:
        answer_idx=relation_types.index(rel)
        if rel=="PIP":
            answer_idx=PIP_index
    else:
        answer_idx=-1
    return span1,type1,span2,type2,'\n'.join(options),options[answer_idx],ent1,ent2

def get_relation(ent1,ent2,rels):
    span=ent1[:2]+ent2[:2]
    for rel in rels:
        if span==rel[:4]:
            return rel[-1]
    span=ent2[:2]+ent1[:2]
    for rel in rels:
        if span==rel[:4]:
            return rel[-1]
    return None

def get_span(ent,sent,offset):
    span=" ".join(sent[ent[0]-offset:ent[1]-offset+1])
    return span,ent[-1]

def combine_json_attributes(source,outfile):
    
    dics=json.loads(open(source).read())
    for idx,dic in enumerate(dics):
        
        # resetting relation index, because some entities are removed
        rel_idx_map={}
        offset=0
        for i1, d in enumerate(dics[idx]['entities']):
            dics[idx]['entities'][i1]["type"]=dics[idx]['entities'][i1]["type"].replace('Possible_Drug','Drug')
            if d["type"]=="Assertion" or d["type"] not in entity_type_map2 or d in dics[idx]['entities'][:i1]:
                offset+=1
            rel_idx_map[i1]=i1-offset


        for i2,r in enumerate(dic['relations']):
            if r['type']=='Problem-Assertion':
                h,t=r['head'],r['tail']
                if dic['entities'][h]['type']=='Problem' and dic['entities'][t]['type']=='Assertion':
                    a=1
                elif dic['entities'][t]['type']=='Problem' and dic['entities'][h]['type']=='Assertion':
                    h,t=t,h
                else:
                    assert False,[dic['entities'][h],dic['entities'][t]]
                if dics[idx]['entities'][t]['subtype'] not in ['None','']:
                    dics[idx]['entities'][h]['subtype']=dics[idx]['entities'][t]['subtype']
            dic['relations'][i2]['head']=rel_idx_map[dic['relations'][i2]['head']]
            dic['relations'][i2]['tail']=rel_idx_map[dic['relations'][i2]['tail']]


        dics[idx]['entities']=[d for it,d in enumerate(dics[idx]['entities']) if d["type"]!="Assertion" and d["type"] in entity_type_map2 and d not in dics[idx]['entities'][:it]]
        for j,d in enumerate(dics[idx]['entities']):
            if d['type']=='Problem' and d['subtype']=='':
                dics[idx]['entities'][j]['subtype']='present'
            
            if d['subtype']!='':
                dics[idx]['entities'][j]['type']=dics[idx]['entities'][j]['type']+'-'+d['subtype']
                # if d['type']!='Problem':
                #     print(d['type'])
        dics[idx]['relations']=[d for d in dics[idx]['relations'] if d["type"]!="Problem-Assertion" and d["type"] in relation_type_map2]
        
        # checking if the index matches
        for i2,r in enumerate(dics[idx]['relations']):
            if '-' in r['type']:
                assert dic['entities'][r['head']]['type'].split('-')[0] in r['type'] \
                    and dic['entities'][r['tail']]['type'].split('-')[0] in r['type']
            dics[idx]['relations'][i2]['type']=relation_type_map2[dics[idx]['relations'][i2]['type']]
        for ie,ent in enumerate(dics[idx]['entities']):
            dics[idx]['entities'][ie]["ent_idx"]=ie
            dics[idx]['entities'][ie]['type']=dics[idx]['entities'][ie]['type'].replace('Characteristics-mild','Severity-mild')
    with open(outfile,'w') as f:
        json.dump(dics,f,indent=4)

def write_types(entities,relations,type_dic=None):
    if not type_dic:
        type_dic={"entities":{},"relations":{}}
    
    for entity in entities:
        type_dic["entities"][entity]= {"short": entity,
                                    "verbose": entity}
    for relation_type in relations:
        type_dic["relations"][relation_type]= {"short": relation_type,
                                        "verbose": relation_type,
                                        "symmetric": False}

    return type_dic

def write_json(current_id,sentences,ners,relations):
    assert len(sentences)==len(ners)
    assert len(sentences)==len(relations)
    return {"id":current_id,
                    "sentences":sentences,
                    "ner":ners,
                    "relations":relations}

def reset_input():

    return 0,[],[],[]

def find_token_information(text,tokenizer):
    doc=nlp(text)
    sents=[sent.start_char for sent in doc.sents]
    # for sent in doc.sents:
    #     print(sent)
    total_sent=0
    tokens=[{'text':tok.text,'start':tok.idx,'end':tok.idx+len(tok.text)} for tok in doc if len(tok.text)>0]
    for idx, tok in enumerate(tokens):
        tokens[idx]['sent_id']=sum([s<=tok['start'] for s in sents])-1
        total_sent=max(total_sent,tokens[idx]['sent_id'])
    
    sent_lengths=[get_token_length(sent.text,tokenizer) for sent in doc.sents]
    assert len(sent_lengths)==total_sent+1
    return tokens,sent_lengths

def get_token_length(text,tokenizer):
    return len(tokenizer(text)['input_ids'])

def brat2spert_note(source_dir,outfile,tokenizer):
    output=[]
    for file in tqdm(glob(source_dir)):
        text=open(file.replace('.ann','.txt')).read()
        tokens,sent_lengths=find_token_information(text,tokenizer)

        lines=[l for l in open(file).read().split('\n') if l]
        entity_type_map={}
        entities=[]
        mapped_entities=[]
        for l in lines:
            if l[0]=='T':
                tb=l.split()[0]
                words=l.split('\t')[1].split()
                type,s,e=words[0],int(words[1]),int(words[-1])
                s=[idx for idx,tok in enumerate(tokens) if s>=tok['start'] and s<tok['end']]
                e=[idx for idx,tok in enumerate(tokens) if e>tok['start'] and e<=tok['end']]
                if len(s)==1 and len(e)==1:
                    s,e=s[0],e[0]+1
                else:
                    #print(l,[s,e])
                    continue
                if type not in Entity_types:
                    #print(l,[s,e])
                    continue
                if e<=s:
                    print(l,[s,e])
                if [s,e,type] in mapped_entities:
                    entity_type_map[tb]=mapped_entities.index([s,e,type])
                    continue
                mapped_entities.append([s,e,type])
                entity_type_map[tb]=len(entities)
                entities.append({
                    'type':type,
                    'start':s,
                    'subtype':'',
                    'end':e,
                    'ent_idx':len(entities),
                    'text':text[tokens[s]['start']:tokens[e-1]['end']],
                    'sent_idx':tokens[s]['sent_id']
                })
        
        for l in lines:
            if l[0]=='A':
                words=l.split()
                assert len(words)==4
                if words[2] in entity_type_map and words[-1] in Entity_subtypes:
                    entities[entity_type_map[words[2]]]['subtype']=words[-1]
                    #print(words[-1])

        relations=[]
        for l in lines:
            if l[0]=='E':
                words=l.replace(":",' ').split()
                etb,tb=words[0],words[2]
                if tb in entity_type_map:
                    entity_type_map[etb]=entity_type_map[tb]
                attributes=l.split()[1:]
                if len(attributes)>1:
                    head=attributes[0].split(':')[1]
                    hidx=entity_type_map[head]
                    if entities[hidx]['type']!='Problem':
                        continue
                    for tail in attributes[1:]:                        
                        tail=tail.split(':')[1]
                        if tail in entity_type_map:
                            tidx=entity_type_map[tail]
                            if tidx!=hidx:
                                ent_type=entities[hidx]['type']+'-'+entities[tidx]['type']
                                relations.append({
                                    'type':ent_type,
                                    'head':hidx,
                                    'tail':tidx,
                                    'head_text':entities[hidx]['text'],
                                    'tail_text':entities[tidx]['text']
                                })
                        else:
                            print(file,tail)
        identified_relations=[]
        
        for l in lines:
            if l[0]=='R':   
                words=l.replace("Arg1:",'').replace("Arg2:",'').split()
                assert len(words)==4,words
                if not words[2] in entity_type_map and words[3] in entity_type_map:
                    print('relation error')
                    continue
                if words[1] in relation_type_map and words[2] in entity_type_map and words[3] in entity_type_map:
                    hidx=entity_type_map[words[2]]

                    tidx=entity_type_map[words[3]]

                    first_sent_idx=min(entities[hidx]['sent_idx'],entities[tidx]['sent_idx'])
                    second_sent_idx=max(entities[hidx]['sent_idx'],entities[tidx]['sent_idx'])

                    st=tokens[entities[hidx]['start']]
                    et=tokens[entities[tidx]['start']]

                    head=entities[hidx]
                    head_s,head_e=tokens[head['start']]['start'],tokens[head['end']-1]['end']
                    tail=entities[tidx]
                    tail_s,tail_e=tokens[tail['start']]['start'],tokens[tail['end']-1]['end']
                    if st['start']>et['start'] and words[1]!='PIP':
                        st,et=et,st
                    if [relation_type_map[words[1]],hidx,tidx] not in identified_relations:
                        identified_relations.append([relation_type_map[words[1]],hidx,tidx])
                        ta=min(first_sent_idx,second_sent_idx)
                        tb=max(first_sent_idx,second_sent_idx)
                        sent_start=[t for t in tokens if t['sent_id']>=first_sent_idx][0]['start']
                        sent_end=[t for t in tokens if t['sent_id']>second_sent_idx]
                        if not sent_end:
                            sent_end=None
                        else:
                            sent_end=sent_end[0]['start']
                        if entities[tidx]['type']=='Drug':
                            hidx,tidx=tidx,hidx
                        relations.append({
                            'type':relation_type_map[words[1]],
                            'head':hidx,
                            'tail':tidx,
                            'head_text':entities[hidx]['text'],
                            'tail_text':entities[tidx]['text'],
                            'sent_distance':abs(second_sent_idx-first_sent_idx),
                            'context_size':sum(sent_lengths[ta:tb+1]),
                            'total token length':sum([sent_lengths[i] for i in range(first_sent_idx,second_sent_idx+1)]),
                            'distance':text[sent_start:sent_end],#text[st['start']:et['end']],
                            'id':(relation_type_map[words[1]],head_s,head_e,tail_s,tail_e)
                        })

        output.append({
            'id':file.split('/')[-1].split('.')[0],
            'tokens':copy.deepcopy(tokens),
            'sent_length':copy.deepcopy(sent_lengths),
            'entities':copy.deepcopy(entities),
            'relations':copy.deepcopy(relations),
        })
        assert len(sent_lengths)==max([t['sent_id'] for t in tokens])+1, [len(sent_lengths),max([t['sent_id'] for t in tokens])+1]
    with open(outfile,'w') as f:
        json.dump(output,f,indent=4)
    return

def spertNote2spertWindow_AllRE(source_file,outfile,max_tokens=400,include_events=False):
    # this code include all relations within this window
    # if setting include_events to the False, will exclude all trigger-attributes connections, and remove all attributes labels
    
    source=json.load(open(source_file))
    final_output=[]

    for data in tqdm(source):
        tokens=data['tokens']
        sent_starts=[0] + [i for i,t in enumerate(tokens) if i>0 and t['sent_id']!=tokens[i-1]['sent_id']]
        sent_ends=[i+1 for i,t in enumerate(tokens) if i<len(tokens)-1 and t['sent_id']!=tokens[i+1]['sent_id']]+[len(tokens)]
        
        current_end=-1
        for start_sent_id,start_token in enumerate(sent_starts):
            end_sent_id=max(current_end+0,start_sent_id)
            end_token=sent_ends[end_sent_id] if end_sent_id<len(sent_ends) else len(tokens)-1

            while end_sent_id<=len(sent_ends) and \
                sum([data["sent_length"][ls] for ls in range(start_sent_id,end_sent_id)])<=max_tokens:
                end_sent_id+=1
                end_token=sent_ends[end_sent_id] if end_sent_id<len(sent_ends) else len(tokens)-1
            
            end_sent_id-=1
            end_token=sent_ends[end_sent_id] if end_sent_id<len(sent_ends) else len(tokens)-1
            end_token=min(end_token,len(tokens)-1)

            # if the current end sentence has not been covered.
            if current_end!=end_sent_id and end_sent_id>=start_sent_id:
                entity_idx=[idx for idx,ent in enumerate(data["entities"]) \
                        if ent['start']>=start_token and ent['start']<end_token \
                            and ent['end']>start_token and ent['end']<=end_token and \
                                (include_events or ent['type'] in TRIGGERS or PROBLEM in ent['type'])]
                
                
                entities=[]
                for i,idx in enumerate(entity_idx):
                    entities.append(copy.deepcopy(data['entities'][idx]))
                    entities[-1]['start']-=start_token
                    entities[-1]['end']-=start_token
                    entities[-1]["ent_idx"]=i
                
                relations=[]
                for rel in data['relations']:
                    if rel['head'] in entity_idx and rel['tail'] in entity_idx:
                        relations.append(copy.deepcopy(rel))
                        relations[-1]['head']=entity_idx.index(rel['head'])
                        relations[-1]['tail']=entity_idx.index(rel['tail'])
                        head=entities[relations[-1]['head']]
                        head_s,head_e=tokens[head['start']+start_token]['start'],tokens[head['end']-1+start_token]['end']
                        tail=entities[relations[-1]['tail']]
                        tail_s,tail_e=tokens[tail['start']+start_token]['start'],tokens[tail['end']-1+start_token]['end']
                        relations[-1]['rel-id']=(rel['type'],head_s,head_e,tail_s,tail_e)
                
                
                final_output.append({
                    'id':'{}-{}-{}'.format(data['id'],tokens[start_token]["start"],tokens[end_token]["start"]),
                    'start_send_id':start_sent_id,
                    'num_sent':end_sent_id-start_sent_id,
                    'offset':tokens[start_token]["start"],
                    'token_end':tokens[end_token]["start"],
                    'sent':' '.join([t['text'] for t in tokens[start_token:end_token]]),
                    'num_BERTtokens':sum([data["sent_length"][ls] for ls in range(start_sent_id,end_sent_id)]),
                    'tokens':[t['text'] for t in tokens[start_token:end_token]],
                    'token_offsets':copy.deepcopy(tokens[start_token:end_token]),
                    'entities':copy.deepcopy(entities),
                    'relations':copy.deepcopy(relations),
                })
    with open(outfile,'w') as f:
        json.dump(final_output,f,indent=4)
    return

def spertNote2spertWindow(split,source_file,outfile,max_tokens=400,
                          include_events=False,include_entities_side=False,max_sent=4,tokenizer=None):
    # this function only include the relations at the beginning and ending sentence of this window.
    # if setting include_events to the False, will exclude all trigger-attributes connections, and remove all attributes labels
    
    source=json.load(open(source_file))
    final_output=[]

    for data in tqdm(source):
        tokens=data['tokens']
        sent_starts=[0] + [i for i,t in enumerate(tokens) if i>0 and t['sent_id']!=tokens[i-1]['sent_id']]
        sent_ends=[i+1 for i,t in enumerate(tokens) if i<len(tokens)-1 and t['sent_id']!=tokens[i+1]['sent_id']]+[len(tokens)]
        

        for start_sent_id,start_token in enumerate(sent_starts):
            end_sent_id=start_sent_id
            end_token=sent_ends[end_sent_id] if end_sent_id<len(sent_ends) else len(tokens)

            while end_sent_id<=min(len(sent_ends),start_sent_id+max_sent) and \
                sum([data["sent_length"][ls] for ls in range(start_sent_id,end_sent_id)])<=max_tokens:
                current_sent=' '.join([t['text'] for t in data["tokens"][start_token:end_token]])
                
                if split=='train' and include_entities_side:
                    # all entities within the window
                    entity_idx=[idx for idx,ent in enumerate(data["entities"]) \
                        if ent['start']>=start_token and ent['start']<end_token \
                            and ent['end']>start_token and ent['end']<=end_token and \
                                (include_events or ent['type'] in TRIGGERS or PROBLEM in ent['type'])]
                else:
                    entity_idx=[idx for idx,ent in enumerate(data["entities"]) \
                        if ent["sent_idx"] in [end_sent_id,start_sent_id] and \
                        ent['start']>=start_token and ent['start']<end_token \
                            and ent['end']>start_token and ent['end']<=end_token and \
                                (include_events or ent['type'] in TRIGGERS or PROBLEM in ent['type'])]

                
                entities=[]
                find_problem=False
                for i,idx in enumerate(entity_idx):
                    entities.append(copy.deepcopy(data['entities'][idx]))
                    entities[-1]['start']-=start_token
                    entities[-1]['end']-=start_token
                    entities[-1]["ent_idx"]=i
                    if PROBLEM in entities[-1]['type']:#==PROBLEM:
                        find_problem=True
                
                #must have relations
                if len(entities)>1 and find_problem:              
                    relations=[]
                    for rel in data['relations']:
                        if rel['head'] in entity_idx and rel['tail'] in entity_idx:
                            if split=='train' or start_sent_id==end_sent_id or \
                                data['entities'][rel['head']]['sent_idx']!=data['entities'][rel['tail']]['sent_idx']:
                                relations.append(copy.deepcopy(rel))
                                relations[-1]['head']=entity_idx.index(rel['head'])
                                relations[-1]['tail']=entity_idx.index(rel['tail'])
                                head=entities[relations[-1]['head']]
                                head_s,head_e=tokens[head['start']+start_token]['start'],tokens[head['end']-1+start_token]['end']
                                tail=entities[relations[-1]['tail']]
                                tail_s,tail_e=tokens[tail['start']+start_token]['start'],tokens[tail['end']-1+start_token]['end']
                                
                                relations[-1]['rel-id']=(rel['type'],head_s,head_e,tail_s,tail_e)
                                       
                    final_output.append({
                        'id':'{}-{}-{}'.format(data['id'],tokens[start_token]["start"],tokens[end_token-1]["start"]),
                        'start_send_id':start_sent_id,
                        'num_sent':end_sent_id-start_sent_id,
                        'offset':tokens[start_token]["start"],
                        'token_end':tokens[end_token-1]["start"],
                        'sent':' '.join([t['text'] for t in tokens[start_token:end_token]]),
                        'num_BERTtokens':sum([data["sent_length"][ls] for ls in range(start_sent_id,end_sent_id)]),
                        'tokens':[t['text'] for t in tokens[start_token:end_token]],
                        'token_offsets':copy.deepcopy(tokens[start_token:end_token]),
                        'entities':copy.deepcopy(entities),
                        'relations':copy.deepcopy(relations),
                    })

                end_sent_id+=1
                end_token=sent_ends[end_sent_id] if end_sent_id<len(sent_ends) else len(tokens)-1

        #break
    with open(outfile,'w') as f:
        json.dump(final_output,f,indent=4)
    return

def spertNote2spertSent(source_file,outfile,max_tokens=400):

    source=json.load(open(source_file))
    final_output=[]

    for data in tqdm(source):
        tokens=data['tokens']
        sent_starts=[0] + [i for i,t in enumerate(tokens) if i>0 and t['sent_id']!=tokens[i-1]['sent_id']]
        sent_ends=[i+1 for i,t in enumerate(tokens) if i<len(tokens)-1 and t['sent_id']!=tokens[i+1]['sent_id']]+[len(tokens)]
        

        for sent_id,start_token in enumerate(sent_starts):
            end_sent_id=sent_id
            end_token=sent_ends[sent_id] if end_sent_id<len(sent_ends) else len(tokens)

            if end_token-start_token<=max_tokens:

                entity_idx=[idx for idx,ent in enumerate(data["entities"]) \
                        if ent["sent_idx"] ==sent_id and \
                            ent['start']>=start_token and ent['start']<end_token \
                            and ent['end']>start_token and ent['end']<=end_token]
                
                
                entities=[]
                for i,idx in enumerate(entity_idx):
                    entities.append(copy.deepcopy(data['entities'][idx]))
                    entities[-1]['start']-=start_token
                    entities[-1]['end']-=start_token
                    entities[-1]["ent_idx"]=i
                    if entities[-1]['start']>=entities[-1]['end']:
                        print(entities[-1])
          
                relations=[]
                for rel in data['relations']:
                    if rel['head'] in entity_idx and rel['tail'] in entity_idx:
                        relations.append(copy.deepcopy(rel))
                        relations[-1]['head']=entity_idx.index(rel['head'])
                        relations[-1]['tail']=entity_idx.index(rel['tail'])
                        head=entities[relations[-1]['head']]
                        head_s,head_e=tokens[head['start']+start_token]['start'],tokens[head['end']-1+start_token]['end']
                        tail=entities[relations[-1]['tail']]
                        tail_s,tail_e=tokens[tail['start']+start_token]['start'],tokens[tail['end']-1+start_token]['end']
                        
                        relations[-1]['rel-id']=(rel['type'],head_s,head_e,tail_s,tail_e)
                                    
                final_output.append({
                    'id':'{}-{}-{}'.format(data['id'],tokens[start_token]["start"],tokens[end_token-1]["start"]),
                    'start_send_id':sent_id,
                    'num_sent':0,
                    'offset':tokens[start_token]["start"],
                    'token_end':tokens[end_token-1]["start"],
                    'sent':' '.join([t['text'] for t in tokens[start_token:end_token]]),
                    'num_BERTtokens':data["sent_length"][sent_id],
                    'tokens':[t['text'] for t in tokens[start_token:end_token]],
                    'token_offsets':copy.deepcopy(tokens[start_token:end_token]),
                    'entities':copy.deepcopy(entities),
                    'relations':copy.deepcopy(relations),
                })
        #break
    with open(outfile,'w') as f:
        json.dump(final_output,f,indent=4)
    return

def spert2plmarker(source_file,output_filename,tokenizer,include_all_relations=True):
    source=json.load(open(source_file))
    final_output=[]

    id_key='orig_id' if 'orig_id' in source[0] else "id"
    current_id=source[0][id_key] 
    offset,sentences,ners,relations=reset_input()

    all_types=[]
    for data in source:
        #write output
        # assumption:in the SpERT input format, the sentences from the same document have the same ids.
        if current_id!=data[id_key]:
            final_output.append(write_json(current_id,sentences,ners,relations))
            current_id=data[id_key]
            offset,sentences,ners,relations=reset_input()
        
        if tokenizer is not None and get_token_length(' '.join(data["tokens"]),tokenizer)>=400:
            print('long sentence')
            continue        
        #check the spert tail index
        assert all([entity["end"]<=len(data["tokens"]) for entity in data["entities"] if entity["type"] in entity_type_map2]), [(entity["end"],len(data["tokens"])) for entity in data["entities"] if entity["type"] in entity_type_map2]
        assert all([data["entities"][rl["head"]]["end"]<=len(data["tokens"]) for rl in data["relations"] if rl["type"] in relation_type_map2])
        assert all([data["entities"][rl["tail"]]["end"]<=len(data["tokens"]) for rl in data["relations"] if rl["type"] in relation_type_map2]) 
        
        sentences.append(data["tokens"])
        ners.append([])
        for idx,entity in enumerate(data["entities"]):
             if entity["type"] in entity_type_map2:
                  etype=entity_type_map2[entity["type"]]
                  #print(entity)
                  if 'subtype' in entity and entity['subtype']!='' and '-' not in etype:
                       etype+=f"-{entity['subtype']}"
                  etype=etype.replace("Characteristics-mild","Severity-mild")
                  if etype not in all_types:
                       all_types.append(etype)  
                  #assert entity["end"]-entity["start"]<=20, [entity,entity["end"]-entity["start"]]
                  assert entity["end"]-1 < len(data["tokens"]), [entity["end"]-1,len(data["tokens"])]
                  ners[-1].append([entity["start"]+offset,entity["end"]+offset-1,etype])
                  data["entities"][idx]['type']=etype
             else:
                 ttt=1
                 #print(entity)
        temp_rl=[]
        for rl in data["relations"]:
            if include_all_relations or rl["type"] in relation_type_map2: #relation_type_map2: # include relations only
                rl["type"]=relation_type_map2[rl["type"]]#,rl["type"])
                head,tail=rl["head"],rl["tail"]

                if data["entities"][tail]["type"] in ["Drug","treatment"]:
                     head,tail=tail,head
                if data["entities"][head]["type"] in entity_type_map2 \
                    and data["entities"][tail]["type"] in entity_type_map2: 
                    s1,e1=data["entities"][head]["start"]+offset,data["entities"][head]["end"]+offset-1
                    s2,e2=data["entities"][tail]["start"]+offset,data["entities"][tail]["end"]+offset-1
                    if [s1,e1,entity_type_map2[data["entities"][head]["type"]]] in ners[-1] and \
                        [s2,e2,entity_type_map2[data["entities"][tail]["type"]]] in ners[-1]:
                        if include_all_relations or rl["type"] in relation_type_map:
                            temp_rl.append([s1,e1,s2,e2,rl["type"]])
                    else:
                        print("errors, the relations are not in the relation map")
                        print([s1,e1,data["entities"][head]["type"]])        
        relations.append(temp_rl)
        offset+=len(data["tokens"])

        #sanity check for input length
        temp=np.sum([len(s) for s in sentences])
        assert offset<=temp,[offset,temp,current_id,data["sent_id"]]

    #adding the final sentences
    final_output.append(write_json(current_id,sentences,ners,relations))    
    
    #write the output
    with open(output_filename,"w",encoding="utf-8") as f:
        for t in final_output:
                json.dump(t,f)
                f.write("\n")
    print("{} notes converted from {} sentences.".format(len(final_output),len(source)))
    all_types.sort()
    print(f'entities {len(all_types)}--------------------')
    #print(all_types)
    print('" ,"'.join(all_types))
    r_types=[]
    r_types=list(set([v for k,v in relation_type_map2.items()]))
    r_types.sort()
    print(f'relations {len(r_types)}--------------------')
    print('" ,"'.join(r_types))
    return all_types, r_types

def plmarker2T5QA(source,out_dir):
    source_lines=open(source).read().split("\n")
    output=[]
    for line in tqdm(source_lines):
        if line:
            dic=json.loads(line)
            id=dic['id']
            offset=0
            for sent,ner,rels in zip(dic["sentences"],dic["ner"],dic["relations"]):
                if len(ner)>=2:
                    for e1 in range(1,len(ner)):
                        for e2 in range(e1):
                            
                            ent1,ent2=ner[e1],ner[e2]
                            rel=get_relation(ent1,ent2,rels)
                            span1,type1=get_span(ent1,sent,offset)
                            span2,type2=get_span(ent2,sent,offset)
                            type1=type1.split('-')[0]
                            type2=type2.split('-')[0]
                            span1,type1,span2,type2,options,answer,ent1,ent2=generate_options(span1,type1,span2,type2,rel,ent1,ent2)
                            
                            if options is not None:
                                output_sent=copy.deepcopy(sent)
                                output_sent[ent1[0]-offset]='<'+output_sent[ent1[0]-offset]
                                output_sent[ent1[1]-offset]=output_sent[ent1[1]-offset]+f'>({type1})'
                                output_sent[ent2[0]-offset]='<'+output_sent[ent2[0]-offset]
                                output_sent[ent2[1]-offset]=output_sent[ent2[1]-offset]+f'>({type2})'
                                output_sent=' '.join(output_sent)

                                assert type2==PROBLEM and type1 in [PROBLEM,DRUG],[type1,type2]
                                output.append({
                                    "id":f"{id}-offset{offset}-<{span1}>({type1}){ent1[0]}-<{span2}>({type2}){ent2[0]}".replace('/home/velvinfu/data/n2c2_2010/',''),
                                    "instruction":f"What is the relationship between <{span1}>({type1}) and <{span2}>({type2}) in this following clinical notes? The relation must be explicitly stated.",
                                    "input":f"Clinical Note: '{output_sent}'\n Options:\n{options}\nAnswer:",
                                    "output":answer,
                                    'relation_source':rel if rel else 'None'
                                })
                            # else:
                            #     print([span1,type1,span2,type2])
                offset+=len(sent)

    with open(out_dir,"w") as f:
        json.dump(output,f,indent=4)

def spert2T5QA(source,out_dir,neg_prob=1):
    source=json.loads(open(source).read())

    output=[]
    ids=[]
    for dic in tqdm(source):
        offset=0
        sent,ner,rels=dic["tokens"],dic["entities"],dic["relations"]
        id=dic['id']
        if len(ner)>=2:
            for e1 in range(1,len(ner)):
                for e2 in range(e1):
                    PIP_index=0
                    if dic['num_sent']!=0 and ner[e1]['sent_idx']==ner[e2]['sent_idx']:
                        continue
                    # if ner[e1]['start']>ner[e2]['start']:
                    #     e1,e2=e2,e1
                        
                    ent1,ent2=ner[e1],ner[e2]
                    ent1=[ent1['start'],min(ent1['end'],len(sent))-1,ent1['type']]
                    ent2=[ent2['start'],min(ent2['end'],len(sent))-1,ent2['type']]
                    rel=[r for r in rels if (r['head']==e1 and r['tail']==e2) or (r['head']==e2 and r['tail']==e1)]
                    
                    if rel:
                        PIP_index=1 if rel[0]['head']==e2 else 0
                        rel=rel[0]['type']
                    else:
                        rel=None
                    span1,type1=get_span(ent1,sent,offset)
                    span2,type2=get_span(ent2,sent,offset)
                    type1=type1.split('-')[0]
                    type2=type2.split('-')[0]
                    span1,type1,span2,type2,options,answer,ent1,ent2=generate_options(span1,type1,span2,type2,rel,ent1,ent2,PIP_index)
                    
                    if options is not None:
                        if dic['num_sent']!=0 and rel is None and random.uniform(0, 1) > neg_prob:
                            continue
                        output_sent=copy.deepcopy(sent)
                        output_sent[ent1[0]-offset]='<'+output_sent[ent1[0]-offset]
                        output_sent[ent1[1]-offset]=output_sent[ent1[1]-offset]+f'>({type1})'
                        output_sent[ent2[0]-offset]='<'+output_sent[ent2[0]-offset]
                        output_sent[ent2[1]-offset]=output_sent[ent2[1]-offset]+f'>({type2})'
                        output_sent=' '.join(output_sent)

                        tid=f"{id}-<{span1}>({type1}){ent1[0]}-<{span2}>({type2}){ent2[0]}"
                        assert type2==PROBLEM and type1 in [PROBLEM,DRUG],[type1,type2]
                        if tid in ids:
                            continue
                        output.append({
                            "id":tid,
                            'num_sent':dic['num_sent'],
                            "instruction":f"What is the relationship between <{span1}>({type1}) and <{span2}>({type2}) in this following clinical notes?",
                            "input":f"Clinical Note: '{output_sent}'\n Options:\n{options}\nAnswer:",
                            "output":answer,
                            'relation_source':rel if rel else 'None'
                        })
                        ids.append(tid)

    with open(out_dir,"w") as f:
        json.dump(output,f,indent=4)
    print(len(output))
    return
def find_entity_idx(s,e,entity_map):
    for k,v in entity_map.items():
        if k[0]==s and k[1]==e:
            return v
    return None

def plmarkerPred2json(source_file,pred_ent_file,pred_re_file,out_file):
    source=[json.loads(l) for l in open(source_file).read().split('\n') if l]
    pred_ent=json.loads(open(pred_ent_file).read())['pred']
    pred_re=json.loads(open(pred_re_file).read())

    output=[]
    offset={}
    for it,dic in enumerate(source):
        if str(it) in pred_ent and str(it) in pred_re:
            entities=pred_ent[str(it)]
            relations=[r for tr in pred_re[str(it)] for r in tr[1]]

            if dic['id'] not in offset:
                offset[dic['id']]=0
            for tokens in dic["sentences"]:
                out_entities,out_relations=[],[]
                entity_map={}
                for s,e,type in entities:
                    if s>=offset[dic['id']] and s<offset[dic['id']]+len(tokens) and \
                        e>=offset[dic['id']] and e<=offset[dic['id']]+len(tokens):
                        entity_map[(s,e,type)]=[len(out_entities),
                                                " ".join(tokens[s-offset[dic['id']]:e+1-offset[dic['id']]]),
                                                type]
                        out_entities.append({
                            'start':s-offset[dic['id']],
                            'end':e+1-offset[dic['id']],
                            'type':type,
                            'subtype':'' if '-' not in type else type.split('-')[-1]
                        })
                for [s1,e1],[s2,e2],r in relations:
                    h,t=find_entity_idx(s1,e1,entity_map),find_entity_idx(s2,e2,entity_map)
                    if h is not None and t is not None:
                        if h[-1]=='Drug':
                            h,t=t,h
                        (h,ht,h_type),(t,tt,t_type)=h,t
                        out_relations.append({
                            'type':r,
                            'head':h,
                            'tail':t,
                            'head_text':ht,
                            "tail_text":tt,
                        })
                    # else:
                    #     print([s1,e1,s2,e2,r,h,t,entities])
                output.append({
                    'id':dic['id'],
                    'offset_word':offset[dic['id']],
                    'tokens':tokens,
                    'entities':copy.deepcopy(out_entities),
                    'relations':copy.deepcopy(out_relations),
                })
                offset[dic['id']]+=len(tokens)
    with open(out_file,'w') as f:
        json.dump(output,f,indent=4)

def spert2GLM_evemts(infile,source_dir,outfile,split):
    sents=json.loads(open(infile).read())

    output=[]

    for idx,sent in enumerate(sents):
        if "orig_id" not in sent:
            sent['orig_id']=sent['id'].split('-')[0]

        start,end=sent["token_offsets"][0]["start"],sent["token_offsets"][-1]["end"]
        text=f'{source_dir}/{sent["orig_id"]}.txt'
        text=open(text).read()[start:end]
        id=f'{split}/{sent["orig_id"]}-{start}to{end}'
        # get entities
        entities=[]
        for idx,ent in enumerate(sent['entities']):
            assert sent['tokens'][ent['start']] in text and sent['tokens'][ent['end']-1] in text, [id,ent]
            if '-' in ent['type']:
                ent["type"],ent['subtype']=ent["type"].split('-')
            # token_offset - sent_offset
            ent_s=sent['token_offsets'][ent['start']]["start"]-sent['token_offsets'][0]["start"]
            ent_e=sent['token_offsets'][ent['end']-1]["end"]-sent['token_offsets'][0]["start"]

            entities.append([ent["type"],ent['subtype'],ent['start'],ent['end'],text[ent_s:ent_e],idx])
        
        entities=sorted(entities,key=lambda x: (x[2],x[3]))

        include=True
        # get problem
        output_text=[]
        for ent in entities:
            if ent[0]=='Drug':
                output_text.append(f"<Drug> {ent[-2]}")
            if ent[0]=='Problem':
                assert ent[1] in ['present','absent','hypothetical','not_patient','possible','conditional']
                values={
                    'Problem':ent[-2],
                    'Assertion':ent[1],
                    'Anatomy':'None',
                    'Duration':'None',
                    'Frequency':'None',
                    'Characteristics':'None',
                    'Change':'None',
                    'Severity':'None',
                }

                for rel in sent['relations']:
                    tar_idx=None
                    if rel['head']==ent[-1]:
                        tar_idx=rel['tail']
                    elif rel['tail']==ent[-1]:
                        tar_idx=rel['head']
                    
                    if tar_idx is not None:
                        rtype=rel['type'].split('-')[-1]
                        if rtype not in values:
                            assert rtype in relation_type_map
                            continue
                        attr=[ent2 for ent2 in entities if ent2[-1]==tar_idx][0]
                        if attr[0] == 'Change':
                            assert attr[1] in ['improving','worsening','no_change','resolved']
                            values[attr[0]]=attr[1]
                        elif attr[0] == 'Severity':
                            assert attr[1] in ['mild','moderate','severe']
                            values[attr[0]]=attr[1]
                        else:
                            if values[attr[0]]=="None":
                                values[attr[0]]=attr[-2]
                            else:
                                assert attr[0] in ["Anatomy",'Characteristics','Duration','Frequency'],attr
                                values[attr[0]]+=" <s> "+attr[-2]

                rs=""
                for key,item in values.items():
                    rs+=f' <{key}> {item.strip()}'
                output_text.append(rs)


        output.append({
            "orig_id":sent["orig_id"],
            "sent_id": sent["start_send_id"],
            "offset": sent["offset"],
            'token_start':start,
            "token_end":end,
            'id':id,
            'instruction':'You are a medical expert. Extract all drug and medical problem events from the following clinical note. All events constraints span-only arguments and/or valued arguments. Span-only arguments must use the span original from the clinical note. A medical problem event contains required arguments as a trigger span and an assertion value (present, absent, possible, conditional, hypothetical, not_patient), as well as optional arguments as at most one anatomy span, at most one duration span, at most one frequency span, characteristics spans, change value (no_change, improving, worsening, resolved), severity value (mild, moderate, severe). The drug event contains a required argument as a trigger span.',
            'input':f"Clinical note: '{text}'",
            'output':' [SEP] '.join(output_text) if output_text else "None"
        })

    with open(outfile,'w') as f:
        json.dump(output,f,indent=4)
    return

def spert2T5_generativeRE(source_file,outfile):
    final_output=[]
    source=json.loads(open(source_file).read())

    for dic in source:      
        tokens=copy.deepcopy(dic['tokens'])

        relations={}
        for k,r in relation_names.items():
            relations[r]=[]
        relation_source=[]
        for rel in dic['relations']:
            if rel['type'] in relation_names:
                h=dic['entities'][rel['head']]
                t=dic['entities'][rel['tail']]
                head=' '.join(tokens[h['start']:h['end']])
                tail=' '.join(tokens[t['start']:t['end']])
                relations[relation_names[rel['type']]].append(f'<head> {head} <tail> {tail}')
                relation_source.append([dic['token_offsets'][h['start']]["start"],dic['token_offsets'][h['end']-1]["end"],
                                        dic['token_offsets'][t['start']]["start"],dic['token_offsets'][t['end']-1]["end"],
                                        rel['type'],
                                        f'<head> {head} <tail> {tail}'])
        
        drug_count,problem_count=0,0
        for ent in dic['entities']:
            if ent['type']=='Drug' or 'Problem' in ent['type']:
                tokens[ent['start']]='<'+tokens[ent['start']]
                tokens[ent['end']-1]+=f">({ent['type']})"
                if ent['type']=='Drug':
                    drug_count+=1
                else:
                    problem_count+=1
        if problem_count>0 and drug_count+problem_count>=2:
            output=[]
            for k,v in relations.items():
                if v:
                    output.append(f'{k}: '+' [SEP] '.join(v))
                else:
                    output.append(f'{k}: None')
            output='\n'.join(output)

            out_id=f"{dic['orig_id']}-offset{dic['offset']}-sent_id{dic['sent_id']}" if 'orig_id' in dic else dic['id']
            final_output.append({
                "id":out_id,
                "instruction":f"Extract all relations related to drug and medical problems from this clinical note. The relation must be explicitly stated.",
                "input":f"Clinical Note: "+' '.join(tokens),
                "output":output,
                'num_sent':dic['num_sent']
                #'relation_source':'[SEP]'.join(relation_source) if relation_source else '',
                })

    with open(outfile,"w") as f:
        json.dump(final_output,f,indent=4)

def write_tb(type,text,event,document_map,output_lines,token_start,id,text_tb=''):
    if type in Sub_types:
        span=text_tb
        subtype=event.replace(f'<{type}>','').strip()
        if subtype not in Sub_types[type]:
            return document_map,output_lines
    else:
        span=event.replace(f'<{type}>','').strip()

    if span in text and '\n' not in span and span!="None" and span!="":
        s,e=text.index(span),text.index(span)+len(span)
        s,e=s+token_start,e+token_start
        if type in ['Drug','Problem'] and (s,e,span) in document_map[id]['visited_idx'][type]:
            #print((s,e,span))
            offset=max([t[1] for t in document_map[id]['visited_idx'][type] if t[2]==span])
            # print([t[1] for t in document_map[id]['visited_idx'][type] if t[2]==span])
            #print((offset-token_start,span,text))
            if span not in text[offset-token_start:]:
                return document_map,output_lines
            s=text[offset-token_start:].index(span)
            s,e=s+offset,s+offset+len(span)
        
        
        document_map[id]['visited_idx'][type].append((s,e,span))
        output_lines.append(f"T{document_map[id]['Tidx']}\t{type} {s} {e}\t{span}")
        if type=='Drug':
            output_lines.append(f"E{document_map[id]['Tidx']}\t{type}:T{document_map[id]['Tidx']}")
        elif type in Sub_types:
            output_lines.append(f"A{document_map[id]['Tidx']}\t{type}Val T{document_map[id]['Tidx']} {subtype}")
            
        document_map[id]['Tidx']+=1
    return document_map,output_lines


def glmEvents2BRAT(pred_file,ann_path,out_dir,ref_dir,ref_file=None):
    if ".jsonl" in pred_file:
        pred={}
        lines = open(pred_file).read().split('\n')
        ref_dic=json.loads(open(ref_file).read())
        for l,rd in zip(lines,ref_dic):
            if l:
                pred[rd["id"]]=json.loads(l)['predict']
    else:   
        pred=json.loads(open(pred_file).read()) 
    
    if os.path.isdir(ref_dir):
        shutil.rmtree(ref_dir)
    os.makedirs(ref_dir)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


    document_map={}
    for key,p_output in pred.items():
        key=key.split('/')[-1]
        id,tok=key.split('-')
        token_start,token_end=tok.split('to')
        token_start,token_end=int(token_start),int(token_end)
        text=f'{ann_path}/{id}.txt'
        text=open(text).read()[token_start:token_end]
        
        if p_output.strip()=="None":
            continue
        
        
        for v in all_attributes+['Problem','Drug']:
            if f"{v}>" in p_output and f"<{v}>" not in p_output:
                p_output=p_output.replace(f"{v}>",f"<{v}>")

        index=len(p_output)
        for v in ['Problem','Drug']:
            if f"<{v}>" in p_output:
                index=min(index,p_output.index(f"<{v}>"))
        p_output=p_output[index:]

        events=[l.strip() for l in p_output.split('[SEP]') if l.strip()]

        if id not in document_map:
            document_map={} ### assuming that all documents are very close, here we clear the memory everytime,
            document_map[id]={
                'Tidx':1,
                'Aidx':1,
                'Eidx':1,
                'Ridx':1,
                'outlines':[],
                'visited_idx':{},
            }
            for t in ["Drug",'Problem'] + all_attributes:
                document_map[id]['visited_idx'][t]=[]


        output_lines=document_map[id]['outlines']

        tbs_map=[] #
        out_entities=[]
        for event in events:
            if '<Drug>' in event:
                document_map,output_lines=write_tb("Drug",text,event,document_map,output_lines,token_start,id)
            elif event.count('<Problem>')==1:
                if all([event.count(f"<{a}>")==1 for a in all_attributes]):
                    for a in all_attributes:
                        event=event.replace(f"<{a}>",f"[SEP]<{a}>")
                    attributes=[l.strip() for l in event.split('[SEP]') if l.strip()]
                    for it,attr in enumerate(attributes):
                        type=attr.replace('<','').replace('>','').split()[0]
                        if type == 'Problem':
                            event=f"E{document_map[id]['Tidx']}\t"
                            text_tb=attributes[0].replace('<Problem>',"").strip()
                            if text_tb not in text:
                                continue
                        if type in ['Problem'] + all_attributes:
                            tid=document_map[id]['Tidx']+0
                            # if '<s>' in attr:
                            #     print(attr)
                            for count,att in enumerate(attr.split('<s>')):
                                document_map,output_lines=write_tb(type,text,att,document_map,output_lines,token_start,id,text_tb=text_tb)
                                if tid!=document_map[id]['Tidx']:
                                    if event[-1]!='\t':
                                        event+=" "
                                    count+=1
                                    if count==1:
                                        count=""
                                    event+=f"{type}{count}:T{document_map[id]['Tidx']-1}"
                                    # if count==2:
                                    #     print(event)
                        assert '[SEP]' not in event,[attr,event]
                    if ':' in event:                
                        output_lines.append(event)
        with open(f"{out_dir}/{id}.ann",'w') as f:
            f.write('\n'.join(output_lines))
        shutil.copy(f'{ann_path}/{id}.txt',f"{out_dir}/{id}.txt")
        shutil.copy(f'{ann_path}/{id}.txt',f"{ref_dir}/{id}.txt")
        shutil.copy(f'{ann_path}/{id}.ann',f"{ref_dir}/{id}.ann")
    return


def align_spert_json(source_file,reference_file,outfile):
    pred=json.loads(open(source_file).read())
    source=json.loads(open(reference_file).read())

    for s in tqdm(source):
        s['entities']=[]
        s['relations']=[]
        for p in pred:
            if p['id']==s['id']:
                s['entities']=copy.deepcopy(p['id'])
                break

    with open(outfile,'w') as f:
        json.dump(source,f,indent=4)
    return



