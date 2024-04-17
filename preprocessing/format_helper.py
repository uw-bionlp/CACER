import re
from glob import glob
import random
import json
import numpy as np
import json
from glob import glob
from tqdm import tqdm
import copy
import spacy
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
    tokens=[{'text':tok.text,'start':tok.idx,'end':tok.idx+len(tok.text)} for tok in doc]
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

def spert2plmarker(source_file,output_filename,tokenizer):
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
        
        if get_token_length(' '.join(data["tokens"]),tokenizer)>1000:
            print('long sentence')
            continue        
        #check the spert tail index
        assert all([entity["end"]<=len(data["tokens"]) for entity in data["entities"] if entity["type"] in entity_type_map2]), data
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
            if rl["type"] in relation_type_map2: #relation_type_map2: # include relations only
                rl["type"]=relation_type_map2[rl["type"]]
                head,tail=rl["head"],rl["tail"]

                #because you need to switch on UW, the drug and treatment always come first
                if data["entities"][tail]["type"] in ["Drug","treatment"]:
                     head,tail=tail,head
                if data["entities"][head]["type"] in entity_type_map2 \
                    and data["entities"][tail]["type"] in entity_type_map2: 
                    s1,e1=data["entities"][head]["start"]+offset,data["entities"][head]["end"]+offset-1
                    s2,e2=data["entities"][tail]["start"]+offset,data["entities"][tail]["end"]+offset-1
                    if [s1,e1,entity_type_map2[data["entities"][head]["type"]]] in ners[-1] and \
                        [s2,e2,entity_type_map2[data["entities"][tail]["type"]]] in ners[-1]:
                        if rl["type"] in relation_type_map:
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