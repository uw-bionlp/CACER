from glob import glob
import json
import os
import shutil
import copy
from tqdm import tqdm
import re
from config import *
tempr={
    "admin_for":"TrAP",
    "not_admin_because":"TrNAP",
    "worsens":"TrWP",
    "worsens_or_NotImproving":"TrWP",
    "causes":"TrCP",
    "improves":"TrIP",
    "improves_or_NotWorsening":"TrIP",
    'PIP':'PIP'}
relation_names={
    'TrAP':'AdminFor',
    'TrNAP':'NotAdminBecause',
    'TrCP':'Causes',
    'TrIP':'Improves',
    'TrWP':'Worsens',
    'PIP':'PIP'
}
relation_values=[v for _,v in relation_names.items()]
relation_map={}
for k,v in tempr.items():
    relation_map[k]=relation_names[v]
    relation_map[v]=relation_names[v]
# mapping them to standarized item names.

INTERETED_RELATIONS={}
for k,v in tempr.items():
    INTERETED_RELATIONS[v]=k
INTERETED_RELATIONS   

def write_tb(text,end,start,etype,tidx,aidx,output_lines,document_map,id,subtype=''):
    dic_key=(etype,start,end)
    if '\n' not in text[start:end] and dic_key not in document_map[id]['entities']:
        document_map[id]['entities'][dic_key]=f'T{tidx}'
        output_lines.append(f'T{tidx}\t{etype} {start} {end}\t{text[start:end]}')
        if subtype!='':
            output_lines.append(f'A{aidx}\t{etype}Val T{tidx} {subtype}')
            aidx+=1
        tidx+=1
    return tidx,aidx,output_lines

def json2brat(source_file,ann_path,offset_key,start_key,end_key,pred_file,out_dir):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    temp_pred=json.loads(open(pred_file).read())
    with open(pred_file,'w') as f:
        json.dump(temp_pred,f,indent=4)
    source=json.loads(open(source_file).read())

    pred=[]
    offset=0
    for i,s in enumerate(source):
        if temp_pred[i-offset]['tokens']==source[i]['tokens']:
            pred.append(copy.deepcopy(temp_pred[i-offset]))
        else:
            offset+=1
            pred.append(copy.deepcopy(source[i]))
            pred[-1]['entities']=[]
            pred[-1]['relations']=[]

    # checking base values
    assert len(pred)==len(source),[len(pred),len(source)]
    assert offset+len(temp_pred)==len(source)
    for p,s in zip(pred,source):
        assert p['tokens']==s['tokens']



    document_map={}

    for p,s in zip(pred,source):
        if "orig_id" in s:
            id=s["orig_id"]
        else:
            id=s['id'].split('-')[0]
        if id not in document_map:
            document_map={} ### assuming that all documents are very close, here we clear the memory everytime,
            document_map[id]={
                'Tidx':1,
                'Aidx':1,
                'Eidx':1,
                'Ridx':1,
                'outlines':[],
                'entities':{},
                'relations':[],
                'out_entities':[],
                'out_relations':[],
            }
        

        text=f'{ann_path}/{id}.txt'
        text=open(text).read()

        tbs_map=[] #
        output_lines=document_map[id]['outlines']
        out_entities=[]
        for ent in p['entities']:
            current_entity=''
            start=s[offset_key][ent['start']][start_key]
            end=s[offset_key][ent['end']-1][end_key]
            assert end>start
            if 'Problem-' in ent['type']:
                asserval=ent['type'].split('-')[-1]

                if ("Problem",start,end) not in document_map[id]['entities']:
                    #
                    current_entity=f"E{document_map[id]['Tidx']}\tProblem:T{document_map[id]['Tidx']} Assertion:T{document_map[id]['Tidx']+1}"
                    document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines=write_tb(text,end,start,"Problem",document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines,document_map,id,subtype='')
                    
                    document_map[id]['entities'][("Problem",start,end)]=f"E{document_map[id]['Tidx']-1}"

                    document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines=write_tb(text,end,start,"Assertion",document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines,document_map,id,subtype=asserval)
                else:
                    current_entity=document_map[id]['entities'][("Problem",start,end)]+""
            elif '-' in ent['type']:
                type,subtype=ent['type'].split('-')
                document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines=write_tb(text,end,start,type,document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines,document_map,id,subtype=subtype)
            elif ent['type']=='Drug':
                if ("Drug",start,end) not in document_map[id]['entities']:
                    current_entity=f"E{document_map[id]['Tidx']}\tDrug:T{document_map[id]['Tidx']}"
                    document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines=write_tb(text,end,start,ent['type'],document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines,document_map,id)
                    document_map[id]['entities'][("Drug",start,end)]=f"E{document_map[id]['Tidx']-1}"
                else:
                    current_entity=document_map[id]['entities'][("Drug",start,end)]+""
            else:
                document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines=write_tb(text,end,start,ent['type'],document_map[id]['Tidx'],document_map[id]['Aidx'],output_lines,document_map,id)
                
            out_entities.append(current_entity+'')
            tbs_map.append(f"{ent['type'].split('-')[0]}:T{document_map[id]['Tidx']-1}")

        output_relations=[]
        for r in p['relations']:
            if r['type'] in INTERETED_RELATIONS:
                E1=out_entities[r['head']]
                E2=out_entities[r['tail']]
                if E1 and E2:
                    if 'Problem' not in E1:
                        E1,E2=E2,E1
                    E1=E1.split()[0]
                    E2=E2.split()[0]
                    outr=f"{INTERETED_RELATIONS[r['type']]} Arg1:{E1} Arg2:{E2}"
                    if outr not in document_map[id]['relations']:
                        document_map[id]['relations'].append(outr)
                        output_relations.append(f"R{document_map[id]['Ridx']}\t{outr}")
                        document_map[id]['Ridx']+=1
            else:
                E1=out_entities[r['head']]
                E2=out_entities[r['tail']]
                if (not E1) and E2:
                    E1,E2=E2,E1
                    r['head'],r['tail']=r['tail'],r['head']
                if E1 and not E2 and 'Problem' in E1:
                    type=tbs_map[r['tail']].split(':')[0]
                    count=out_entities[r['head']].count(type)
                    if count>0:
                        tbs_map[r['tail']]=tbs_map[r['tail']].replace(":",f"{count+1}:")
                    out_entities[r['head']]+=' '+tbs_map[r['tail']]
                    #print(out_entities[r['head']])
        
        for l in out_entities+output_relations:
            if l and l not in output_lines:
                if (l[0]=='E' and ':' in l):
                    type,tb=l.replace(':',' ').split()[1:3]
                    output_lines.append(l)
                    if not any([f"{tb}\t{type}" in lt for lt in output_lines]):
                        print(id)
                        print(l)
                        
                if l[0]=='R':
                    output_lines.append(l)

        with open(f"{out_dir}/{id}.ann",'w') as f:
            # output_lines=list(set(output_lines))
            # output_lines.sort()
            f.write('\n'.join(output_lines))
        shutil.copy(f'{ann_path}/{id}.txt',f"{out_dir}/{id}.txt")
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
    counter={}
    for it,dic in enumerate(source):
        if str(it) in pred_ent and str(it) in pred_re:
            entities=pred_ent[str(it)]
            relations=[r for tr in pred_re[str(it)] for r in tr[1]]

            for tokens in dic["sentences"]:
                out_entities,out_relations=[],[]
                entity_map={}
                for s,e,type in entities:
                    if e<len(tokens) and s>=0 and e>=s: 
                        entity_map[(s,e,type)]=[len(out_entities)," ".join(tokens[s:e+1]),type]
                        out_entities.append({
                            'start':s,
                            'end':e+1,
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
                        counter[r]=counter.get(r,0)+1
                    # else:
                    #     print([s1,e1,s2,e2,r,h,t,entities])
                output.append({
                    'id':dic['id'],
                    'offset':dic['id'].split('-')[1],
                    'tokens':tokens,
                    'entities':copy.deepcopy(out_entities),
                    'relations':copy.deepcopy(out_relations),
                })
    with open(out_file,'w') as f:
        json.dump(output,f,indent=4)
    print(counter)
    return

def map_entities(gold, predicted):
    for idx, ent1 in enumerate(predicted):
        for ent2 in gold:
            if ent1[-2]==ent2[-2] and \
                ((ent1[0]>=ent2[0] and ent1[0]<ent2[1]) or (ent1[1]>ent2[0] and ent1[1]<=ent2[1]) \
                 or (ent1[0]<=ent2[0] and ent1[1]>=ent2[1])or (ent1[0]>=ent2[0] and ent1[1]<=ent2[1])):
                predicted[idx]=copy.deepcopy(ent2)
                #break
    return predicted

def read_relations(entities,relations,num_sent,start,end):
    result=[]
    for r in relations:
        if r['type'] in relation_map:
            if num_sent==0 or (entities[r['tail']][-1]!=entities[r['head']][-1] and \
                    entities[r['tail']][-1] in [start,end] and entities[r['head']][-1] in [start,end]):
                tr=[entities[r['head']],entities[r['tail']],relation_map[r['type']]]
                if True or relation_map[r['type']]=='PIP':
                    if entities[r['head']]>=entities[r['tail']]:
                        tr=[entities[r['tail']],entities[r['head']],relation_map[r['type']]]
                if tr and tr not in result:
                    result.append(tr)
    return result

def find_relation(p):
    r=None
    if 'none of' not in p.lower():
        if 'is given as a treatment for' in p and '(A)' in p:
            r='TrAP'
        elif 'is not given or discontinued because' in p:
            r='TrNAP'
        elif 'does not cure' in p or '(C)' in p:
            r='TrWP'
        elif 'but it causes' in p:
            r='TrCP'
        elif 'improves, cures,' in p:
            r='TrIP'
        elif 'causes, describes or reveals' in p:
            r='PIP'
    return r

def read_entities(entities,tokens,start_token_idx=0):
    result=[]
    for ent in entities:
            result.append(([ent['start']+start_token_idx,
                            ent['end']+start_token_idx+1,
                            ent['type'].split('-')[0],
                            tokens[ent['start']+start_token_idx]['sent_id']]))
    return result

def score_relations(pred_file,source_file,out_file,error_file):
    temp_pred=json.loads(open(pred_file).read())
    source=json.loads(open(source_file).read())

    # the extremely long sentence were omited, so using the dummy entities from gold label to replace it.
    pred=temp_pred
    offset=0
    pred=[]
    offset=0
    for i,s in enumerate(source):
        if i-offset<len(temp_pred) and temp_pred[i-offset]['tokens']==source[i]['tokens']:
            pred.append(copy.deepcopy(temp_pred[i-offset]))
        else:
            offset+=1
            pred.append(copy.deepcopy(source[i]))
            pred[-1]['entities']=[]
            pred[-1]['relations']=[]

    # checking base values
    assert len(pred)==len(source),[len(pred),len(source)]
    assert offset+len(temp_pred)==len(source),[offset,len(temp_pred),len(source)]
    for p,s in zip(pred,source):
        assert p['tokens']==s['tokens']
    for re_type in ['intra','inter','any']:
        scores={}
        scores_by_ids={}

        for r in relation_values:
            scores[r]=[0,0,0]

        output_error={}
        count=0
        similarity_map={}
        for r1 in relation_values + ['None']:
            for r2 in relation_values + ['None']:
                similarity_map[(r1,r2)]=0

        for p,s in tqdm(zip(pred,source)):
            if s['entities']==[] and p['entities']==[]:
                continue
            if re_type=='intra' and s['num_sent']!=0:
                continue
            if re_type=='inter' and s['num_sent']==0:
                continue      
            p_entities=read_entities(p['entities'],s["token_offsets"])
            s_entities=read_entities(s['entities'],s["token_offsets"])
            p_entities=map_entities(s_entities, p_entities)

            p_relations=read_relations(p_entities,p['relations'],s['num_sent'],s["start_send_id"],s["start_send_id"]+s['num_sent'])
            s_relations=read_relations(s_entities,s['relations'],s['num_sent'],s["start_send_id"],s["start_send_id"]+s['num_sent'])

            doc_id= s["orig_id"] if 'orig_id' in s else s["id"]
            doc_id=doc_id.split('-')[0]
            if doc_id not in scores_by_ids:
                    scores_by_ids[doc_id]={}
                    for k in relation_values:
                        scores_by_ids[doc_id][k]=[0,0,0]  

            for r in relation_values:
                scores[r][0]+=len([d for d in s_relations if d[-1]==r])
                scores[r][1]+=len([d for d in p_relations if d[-1]==r])
                scores_by_ids[doc_id][r][0]+=len([d for d in s_relations if d[-1]==r])
                scores_by_ids[doc_id][r][1]+=len([d for d in p_relations if d[-1]==r])
            #assert p_entities==s_entities
            FP=[d for d in p_relations if d not in s_relations]
            FN=[d for d in s_relations if d not in p_relations]
            TP=[d for d in s_relations if d in p_relations]
            id_key=s["orig_id"]+'-'+str(s.get('sent_id',"start_send_id")) if 'orig_id' in s else s["id"]+'-'+str(s.get('offset',0))

            if not isinstance(s['tokens'][0],str):
                s['tokens']=[ds['text'] for ds in s['tokens']]
            if not isinstance(p['tokens'][0],str):
                p['tokens']=[ds['text'] for ds in p['tokens']]
            if FP+FN:       
                output_error[id_key]={
                    'tokens':' '.join(s['tokens']),
                    'source_entities':[' '.join(p['tokens'][e[0]:e[1]])+'-'+e[2] for e in s_entities],
                    'pred_entities':[' '.join(p['tokens'][e[0]:e[1]])+'-'+e[2] for e in p_entities],
                    'FP':['{} - {} - {}'.format(' '.join(s['tokens'][l[1][0]:l[1][1]]), l[-1], \
                        ' '.join(s['tokens'][l[0][0]:l[0][1]])) \
                            for l in FP],
                    'FP_dic':FP,
                    'FN':['{} - {} - {}'.format(' '.join(s['tokens'][l[1][0]:l[1][1]]), l[-1], \
                        ' '.join(s['tokens'][l[0][0]:l[0][1]])) \
                            for l in FN],
                    'FN_dic':FN,
                    'TP':['{} - {} - {}'.format(' '.join(s['tokens'][l[1][0]:l[1][1]]), l[-1], \
                        ' '.join(s['tokens'][l[0][0]:l[0][1]])) \
                            for l in TP],
                }
            
            temp_p_relations=copy.deepcopy(p_relations)
            for r1 in s_relations:
                found=False
                for it,r2 in enumerate(temp_p_relations):
                    if r2[:2]==r1[:2]:
                        similarity_map[(r1[-1],r2[-1])]+=1
                        found=True
                        temp_p_relations[it]=[None,None,None]
                        break
                if not found:
                    similarity_map[(r1[-1],'None')]+=1
            
            for r2 in temp_p_relations:
                if r2[-1]==None:
                    continue
                similarity_map[('None',r2[-1])]+=1
            for i,r1 in enumerate(p_relations):
                for j,r2 in enumerate(s_relations):
                    if r1==r2:
                        scores[r1[-1]][-1]+=1
                        scores_by_ids[doc_id][r1[-1]][-1]+=1
                        s_relations[j]='None'
            
            
        overall=[0,0,0]
        for k,v in scores.items():
            for i in range(3):
                overall[i]+=v[i]

        scores['Overall']=overall

        for doc_id in scores_by_ids:
            scores_by_ids[doc_id]['overall']=[0,0,0]
            for k,v in scores_by_ids[doc_id].items():
                for i in range(3):
                    scores_by_ids[doc_id]['overall'][i]+=v[i]

        with open(out_file.replace('.csv',f'_{re_type}.csv'),'w') as f:
            f.write('type,NT,NP,TP,P,R,F1\n')
            for k,[NT,NP,TP] in scores.items():
                P=TP*100/NP if NP else 0
                R=TP*100/NT if NT else 0
                F1=2*P*R/(P+R) if P+R else 0
                f.write(f'{k},{NT},{NP},{TP},{P},{R},{F1}\n')
            
            
            f.write('\n\n\n\n row as GOLD, col as PRED\n')
            f.write(',{}'.format(','.join(relation_values + ['None'])))
            for r1 in relation_values + ['None']:
                f.write(f'\n{r1},')
                for r2 in relation_values + ['None']:
                    f.write(f'{similarity_map[(r1,r2)]},')

        with open(out_file.replace('.csv',f'_{re_type}_by_docs.csv'),'w') as f:
            f.write('doc_id,type,NT,NP,TP,P,R,F1\n')
            for doc_id in scores_by_ids:
                for k,[NT,NP,TP] in scores_by_ids[doc_id].items():
                    P=TP*100/NP if NP else 0
                    R=TP*100/NT if NT else 0
                    F1=2*P*R/(P+R) if P+R else 0
                    f.write(f'{doc_id},{k},{NT},{NP},{TP},{P},{R},{F1}\n')
        with open(error_file.replace('.json',f'_{re_type}.json'),'w') as f:
            json.dump(output_error,f,indent=4)
    return

def extra_rel_values(s):
    # Define the pattern to match the values inside {}
    pattern = r'-<(.+?)>\((.+?)\)(.+?)-<(.+?)>\((.+?)\)(.+?)\[SEP\]'
    
    # Use re.search to find the match
    match = re.search(pattern, s)
    
    # If a match is found, extract the groups
    if match:
        return match.groups()
    else:
        print(s)
        return None
def QARe2json(spert_dir,llama_input_dir,llama_output_dir,outfile):
    if '.jsonl' in llama_output_dir:
        preds=[json.loads(l)['predict'] for l in open(llama_output_dir).read().split('\n') if l.strip()]
    else:
        preds=[v for _,v in json.loads(open(llama_output_dir).read()).items()]

    llama_source=json.loads(open(llama_input_dir).read())
    spert=json.loads(open(spert_dir).read())
    for dic in spert:
        dic['entities']=[]
        dic['relations']=[]

    counter={}
    spert_idx=0
    for i,pred in enumerate(preds):
        pred_rel=find_relation(pred)
        if pred_rel is not None:
            source_id=llama_source[i]['id']
            offset=int(source_id.split('-')[1])
            if spert[spert_idx]['id']!=source_id:
                temp=[idx for idx,d in enumerate(spert) if  d['id']+"-"==source_id[:len(d['id']+"-")]]
                if len(temp)==0:
                    print('match failture')
                    continue
                #assert len(spert_idx)>=1,[len(spert_idx),source_id,spert[spert_idx[0]],spert[spert_idx[1]]]
                spert_idx=temp[0]
            tokens=spert[spert_idx]['tokens']
            
            head_string,head_type,head_idx,tail_string,tail_type,tail_idx=extra_rel_values(source_id+"[SEP]")
            head_idx,tail_idx=int(head_idx),int(tail_idx)
            head=[ent["ent_idx"] for ent in spert[spert_idx]['entities'] \
                if f'-<{" ".join(tokens[ent["start"]:ent["end"]])}>({ent["type"].split("-")[0]}){ent["start"]}-' in source_id]
            if len(head)==0:
                head=len(spert[spert_idx]['entities'])
                head_end=max([i for i in range(head_idx,len(spert[spert_idx]['tokens'])) if ' '.join(spert[spert_idx]['tokens'][head_idx+1:i]) in head_string])
            
                spert[spert_idx]['entities'].append({
                    "type": head_type,
                    "start": head_idx,
                    "subtype": "",
                    "end": head_end,
                    "ent_idx": head,
                    "text": head_string
                })
            else:
                head=head[0]

            tail=[ent["ent_idx"] for ent in spert[spert_idx]['entities'] if ent['ent_idx']!=head and \
                f'-<{" ".join(tokens[ent["start"]:ent["end"]])}>({ent["type"].split("-")[0]}){ent["start"]}[SEP]' in source_id+'[SEP]']
            #assert len(tail)==1,[tail,source_id,spert[spert_idx]['entities']]
            if len(tail)==0:
                tail=len(spert[spert_idx]['entities'])
                tail_end=max([i for i in range(tail_idx,len(spert[spert_idx]['tokens'])) if ' '.join(spert[spert_idx]['tokens'][tail_idx+1:i]) in tail_string])
                spert[spert_idx]['entities'].append({
                    "type": tail_type,
                    "start": tail_idx,
                    "subtype": "",
                    "end": tail_end,
                    "ent_idx": tail,
                    "text": tail_string
                })
            else:
                tail=tail[0]

            spert[spert_idx]['relations'].append({
                'type':pred_rel,
                'tail':head,
                'head':tail
            })
            counter[pred_rel]=counter.get(pred_rel,0)+1
    with open(outfile,'w') as f:
        json.dump(spert,f,indent=4)
    print(counter)
    return

def GenRe2json(spert_dir,llama_input_dir,llama_output_dir,outfile):
    if 't5' in llama_output_dir:
        pred=json.loads(open(llama_output_dir).read())
    else:
        reference=json.loads(open(llama_input_dir).read())
        pred={}
        idx=0
        for l in open(llama_output_dir).read().split('\n'):
            if l.strip():
                dic=json.loads(l)
                pred[reference[idx]['id']]=dic['predict']
                idx+=1

    counter={}
    spert=json.loads(open(spert_dir).read())
    for dic in spert:
        dic['relations']=[]
        if dic['id'] in pred:
            # extracting the predicten entity by type
            for name in relation_names_reversed:
                pred[dic['id']]=pred[dic['id']].replace(f'{name}:',f'\n{name}:')
            pred[dic['id']]=pred[dic['id']].replace(' head>',' <head>').replace(' tail>',' <tail>')
            pred_relations=[l.strip() for l in pred[dic['id']].split('\n') if l.strip()]
            #assert len(pred_relations)==6, pred[dic['id']]
            
            for rel in pred_relations:
                #print(rel)
                for (name,type) in relation_names_reversed.items():
                    start=f'{name}:'
                    if rel[:len(start)]==start:
                        if rel[len(start):].strip().lower()=='none':
                            continue
                        relations=rel[len(start):].split('[SEP]')
                        for relation in relations:
                            if relation.count('<tail>')==1:
                                head,tail=relation.split('<tail>')
                                head=head.replace('<head>','').strip().lower()
                                tail=tail.strip().lower()
                                # finding the entity indx
                                if dic['num_sent']==0:
                                    head=[idx for idx,ent in enumerate(dic['entities']) if ent['text'].lower()==head]
                                    tail=[idx for idx,ent in enumerate(dic['entities']) if ent['text'].lower()==tail]
                                else:
                                    head=[idx for idx,ent in enumerate(dic['entities']) if ent['text'].lower()==head and ent['sent_idx'] in [dic["start_send_id"],dic["start_send_id"]+dic["num_sent"]]]
                                    tail=[idx for idx,ent in enumerate(dic['entities']) if ent['text'].lower()==tail and ent['sent_idx'] in [dic["start_send_id"],dic["start_send_id"]+dic["num_sent"]]]
                                    
                                if head and tail:
                                    dic['relations'].append({
                                        'type':type,
                                        'head':head[0],
                                        'tail':tail[0],
                                        "sent_distance":abs(dic['entities'][head[0]]["sent_idx"]-dic['entities'][tail[0]]["sent_idx"])
                                    })
                                    counter[type]=counter.get(type,0)+1
    with open(outfile,'w') as f:
        json.dump(spert,f,indent=4)
    return

def match_entities(text, tokens):
    result=[]
    if text in tokens:
        return [(tokens.index(text),tokens.index(text)+1)]
    for start_idx,tok in enumerate(tokens):
        if tok==text[:len(tok)]:
            temp_text=tok+''
            for end_idx,tok2 in enumerate(tokens[start_idx+1:]):
                temp_text+=' '+tok2
                if temp_text==text:
                    end_idx+=start_idx+1
                    result.append((start_idx,end_idx+1))
                if temp_text not in text:
                    break
    return result

def insert_entity(start_idx,end_idx,ent_type,dic):
    ent_idx=[idx for idx, ent in enumerate(dic['entities']) if ent['start']==start_idx and ent['end']==end_idx]
    if ent_idx!=[]:
        ent_idx=ent_idx[0]
    else:
        temp_dic={
            'start':start_idx,
            'end':end_idx,
            'type':ent_type,
            'sent_idx':dic['token_offsets'][start_idx]['sent_id']
        }
        dic['entities'].append(copy.deepcopy(temp_dic))
        ent_idx=len(dic['entities'])-1
    return ent_idx,dic
def GenRe2json_predEnt(spert_dir,llama_input_dir,llama_output_dir,outfile):
    
    # read predictions
    if 'llama' not in llama_output_dir:
        pred=json.loads(open(llama_output_dir).read())
    else:
        reference=json.loads(open(llama_input_dir).read())
        pred={}
        idx=0
        for l in open(llama_output_dir).read().split('\n'):
            if l.strip():
                dic=json.loads(l)
                pred[reference[idx]['id']]=dic['predict']
                idx+=1
    # read reference, and store the predictions into the reference file
    spert=json.loads(open(spert_dir).read())
    
    counter={}
    for dic in spert:
        dic['entities']=[]
        dic['relations']=[]
        # if dic['id']!='4575139440-4181-4256-4181':
        #     continue
        if dic['id'] in pred:
            # extracting the predicten entity by type
            for name in relation_names_reversed:
                pred[dic['id']]=pred[dic['id']].replace(f'{name}:',f'\n{name}:')
            pred[dic['id']]=pred[dic['id']].replace(' head>',' <head>').replace(' tail>',' <tail>')
            pred_relations=[l.strip() for l in pred[dic['id']].split('\n') if l.strip()]
            #assert len(pred_relations)==6, pred[dic['id']]
            tokens=[(idx,tok) for idx,tok in enumerate(dic['token_offsets'])\
                        if dic['num_sent']==0 or \
                        tok['sent_id'] in [dic["start_send_id"],dic["start_send_id"]+dic["num_sent"]]
                        ]
            for rel in pred_relations:
                #print(rel)
                for (name,type) in relation_names_reversed.items():
                    start=f'{name}:'
                    if rel[:len(start)]==start:
                        if rel[len(start):].strip().lower()=='none':
                            continue
                        relations=rel[len(start):].split('[SEP]')
                        for relation in relations:
                            if relation.count('<tail>')==1:
                                head,tail=relation.split('<tail>')
                                head=head.replace('<head>','').strip()
                                tail=tail.strip()
                                # finding the entity indx
                                
                                head=match_entities(head, dic['tokens'])
                                tail=match_entities(tail, dic['tokens'])
                                if head and tail:
                                    if type!='PIP':
                                        head_ent='Drug'
                                        #head,tail=tail,head
                                    else:
                                        head_ent='Problem'
                                    #print(head_ent,type)
                                    head_idx,dic=insert_entity(head[0][0],head[0][1],head_ent,dic)
                                    tail_idx,dic=insert_entity(tail[0][0],tail[0][1],'Problem',dic)
                                    dic['relations'].append({
                                        'type':type,
                                        'head':head_idx,
                                        'tail':tail_idx,
                                        "sent_distance":abs(dic['entities'][head_idx]["sent_idx"]-dic['entities'][tail_idx]["sent_idx"])
                                    })
                                    counter[type]=counter.get(type,0)+1
                                # else:
                                #     print(relation,head,tail)

    with open(outfile,'w') as f:
        json.dump(spert,f,indent=4)
    print(counter)

    return