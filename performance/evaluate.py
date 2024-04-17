from glob import glob
import json
import os
import shutil
import copy

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
        
def map_entities(gold, predicted):
    for idx, ent1 in enumerate(predicted):
        for ent2 in gold:
            if ent1[-1]==ent2[-1] and \
                ((ent1[0]>=ent2[0] and ent1[0]<ent2[1]) or (ent1[1]>ent2[0] and ent1[1]<=ent2[1]) \
                 or (ent1[0]<=ent2[0] and ent1[1]>=ent2[1])):
                predicted[idx]=copy.deepcopy(ent2)
                #break
    return predicted

def read_relations(entities,relations):
    result=[]
    for r in relations:
        if r['type'] in relation_map:
            tr=[entities[r['head']],entities[r['tail']],relation_map[r['type']]]
            if relation_map[r['type']]=='PIP':
                if entities[r['head']]>=entities[r['tail']]:
                    tr=[entities[r['tail']],entities[r['head']],relation_map[r['type']]]
            if tr not in result:
                result.append(tr)
    return result


def read_entities(entities):
    result=[]
    for ent in entities:
            result.append(([ent['start'],ent['end'],ent['type'].split('-')[0]]))
    return result

def score_relations(pred_file,source_file,out_file,error_file):
    temp_pred=json.loads(open(pred_file).read())
    source=json.loads(open(source_file).read())

    # the extremely long sentence were omited, so using the dummy entities from gold label to replace it.
    pred=[]
    offset=0
    for i,s in enumerate(source):
        if i-offset<len(temp_pred) and temp_pred[i-offset]['tokens']==source[i]['tokens']:
            pred.append(copy.deepcopy(temp_pred[i-offset]))
        else:
            offset+=1
            pred.append(copy.deepcopy(source[i]))
            pred[-1]['entities']=None
            pred[-1]['relations']=[]

    # checking base values
    assert len(pred)==len(source),[len(pred),len(source)]
    assert offset+len(temp_pred)==len(source),[offset,len(temp_pred),len(source)]
    for p,s in zip(pred,source):
        assert p['tokens']==s['tokens']

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

    for p,s in zip(pred,source):
        if p['entities'] is None:
            continue
        p_entities=read_entities(p['entities'])
        s_entities=read_entities(s['entities'])
        p_entities=map_entities(s_entities, p_entities)

        p_relations=read_relations(p_entities,p['relations'])
        s_relations=read_relations(s_entities,s['relations'])

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
        id_key=s["orig_id"]+'-'+str(s['sent_id']) if 'orig_id' in s else s["id"]+'-'+str(s.get('offset',0))

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
                'FN':['{} - {} - {}'.format(' '.join(s['tokens'][l[1][0]:l[1][1]]), l[-1], \
                       ' '.join(s['tokens'][l[0][0]:l[0][1]])) \
                        for l in FN],
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

    with open(out_file,'w') as f:
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

    with open(out_file.replace('.csv','_by_docs.csv'),'w') as f:
        f.write('doc_id,type,NT,NP,TP,P,R,F1\n')
        for doc_id in scores_by_ids:
            for k,[NT,NP,TP] in scores_by_ids[doc_id].items():
                P=TP*100/NP if NP else 0
                R=TP*100/NT if NT else 0
                F1=2*P*R/(P+R) if P+R else 0
                f.write(f'{doc_id},{k},{NT},{NP},{TP},{P},{R},{F1}\n')
    with open(error_file,'w') as f:
        json.dump(output_error,f,indent=4)
    return