from glob import glob
import shutil
import os
from config import *
import pandas as pd



def remove_invalid(anns):
    out_lines=[]
    removed=[]
    ids_to_remove=[]
    # remove invalid types
    valid_ids=[]
    for line in anns:
        if line[0]=="T" and line.split()[1] not in VALID_TYPES:
            id=line.split()[0]
            removed.append(['invalid argument type',line.split()[1],line])
            ids_to_remove.append(f':{id}')
            continue
        if line[0] in ["T",'R'] and 'Error' in line.split()[1]:
            id=line.split()[0]
            removed.append(['error flag',line.split()[1],line])
            ids_to_remove.append(f':{id}')
            continue   
        if line[0]=='E':
            continue
        valid_ids.append(line.split()[0])
        out_lines.append(line) 
    
    # remove events containing those invalid span types, and remove multipl arguments from drug
    for line in anns:              
        if line[0]=="E":
            if line.split()[1][:4]=="Drug":
                if len(line.split())>2: # drug with multiple arguments
                    removed.append(['removing arguments from drug events','',line])
                    words=line.split()
                    assert words[1][:4]=="Drug"
                    line=words[0]+"\t"+words[1]
            words=line.split()
            new_words=[]
            for w in words[1:]:
                if w.strip() and not any([id==w[-len(id):] for id in ids_to_remove]):
                    new_words.append(w)
                elif w.strip():
                    removed.append(['removing argument from events',w,line])
                elif w.split(':')[1] not in valid_ids:
                    removed.append(['removing span ids without span',w,line])
            line=words[0]+"\t"+' '.join(new_words)
        out_lines.append(line) 
    return out_lines, removed

def merging_labels(anns,removed,text):
    ids_to_replace={}
    outlines=[]

    # idx
    for type in VALID_TYPES:
            text_bounds=[]
            for line in anns:
                if line[0]=='T' and line.split()[1]==type:
                    id=line.split()[0]
                    index=line.split('\t')[1].split()
                    text_bounds.append([int(index[1]),int(index[-1]),id,line])
            text_bounds=sorted(text_bounds,key=lambda x: (x[0],x[1]))
            for idx in range(len(text_bounds)-1):
                s1,e1,id1,l1=text_bounds[idx]
                s2,e2,id2,l2=text_bounds[idx+1]
                if l1==l2:
                    continue
                if (s2>=s1 and s2<e1) or (e2>s1 and e2<=e1):
                    s,e=min(s1,s2),max(e1,e2)
                    span=text[s:e]
                    if '\n' not in span:
                        words=l1.split('\t')
                        l1=f'{words[0]}\t{type} {s} {e}\t{span}'
                    removed.append(('merging spans',f'{l1.split()[0]} & {l2.split()[0]}',f'{l1} & {l2}'))
                    text_bounds[idx+1]=(s1,e1,id1,l1)
                    ids_to_replace[id2]=id1
                outlines.append(l1)
            if text_bounds and (text_bounds[-1][-1]) not in outlines:
                outlines.append(text_bounds[-1][-1])
    
    for line in anns:
        if line[0]=='A':
            for oid,nid in ids_to_replace.items():
                if f' {oid} ' in line:
                    line=line.replace(oid,nid)
            outlines.append(line)


    for line in anns:
        if line[0]=='E':
            words=line.split()
            for idx in range(1,len(words)):
                for oid,nid in ids_to_replace.items():
                    oid=f':{oid}'
                    if oid==words[idx][-len(oid):]:
                        words[idx]=words[idx].split(':')[0]+':'+nid
                        removed.append(('event argument replaced',f'from {oid} to {nid}',line))
            new_words=[]
            current_ids=[]
            for w in words[1:]:
                id=w.split(':')[1]
                if id not in current_ids:
                    new_words.append(w)
                    current_ids.append(id)
            line=words[0]+'\t'+' '.join(new_words)
            outlines.append(line)

    outlines.extend([l for l in anns if l[0]=='R'])
    
    return outlines,removed

def merging_events(anns,removed):
    # merging events with the exact same trigger.
    event_to_replace={}
    outlines=[]
    events={}
    for line in anns:
        if line[0]=='E':
            id,key=line.split()[:2]
            if key in events:
                old_id=events[key].split()[0]
                if len(events[key])>len(line):
                    event_to_replace[id]=old_id
                    continue
                event_to_replace[old_id]=id
                for k,v in event_to_replace.items():
                    if v==old_id:
                        event_to_replace[k]==id
            events[key]=line
        elif line[0] not in ['R','A']:
            outlines.append(line)
    for k,v in events.items():
        outlines.append(v)
    
    # merging relations with the same events, merging assertion with the same values
    for char in ['A','R']:
        relations=[]
        for l in anns:
            if l[0]==char:
                relation='\t'.join(l.split('\t')[1:])+' '
                for oid,nid in event_to_replace.items():
                    relation=relation.replace(f':{oid} ',f':{nid} ')
                relations.append(relation[:-1])
        relations=list(set(relations))
        print(relations)
        outlines.extend([f'{char}{idx}\t{r}' for idx,r in enumerate(relations)])
    
    return outlines,removed


if __name__ == "__main__":

    # source_dir=f"" # the source directory containing all ann files.
    # outdir="" # the output directory to store all the BRAT files with errors.
    # files=glob(f"{source_dir}/*.ann")
    # SUMMARY_FILE="" # the output csv file for a summary of errors.

    split='train'
    source_dir=f"/home/velvinfu/data/ProstateDLBCLPostDiagnosis/R15_dataset_paper/final3/{split}" # the source directory containing all ann files.
    outdir=f"/home/velvinfu/data/ProstateDLBCLPostDiagnosis/R15_dataset_paper/final3/{split}_cleaned" # the output directory to store all the BRAT files with errors.
    print(outdir)
    SUMMARY_FILE=f"/home/velvinfu/data/ProstateDLBCLPostDiagnosis/R15_dataset_paper/final3/{split}_auto.csv" # the output csv file for a summary of errors.

    files=glob(f"{source_dir}/*.ann")
    changes=[]
    os.makedirs(outdir,exist_ok=True)
    for file in files:
        text=open(file.replace('.ann','.txt')).read()
        anns=[l.strip() for l in open(file).read().split('\n') if l.strip()]

        anns, removed=remove_invalid(anns)
        anns=list(set(anns))

        anns, removed=merging_labels(anns,removed,text)
        anns, removed=merging_events(anns,removed)
        
        anns=list(set(anns))
        anns.sort()
        with open(file.replace(source_dir,outdir),'w') as f:
            for char in ['T','A','E','R']:
                f.write('\n'.join([l for l in anns if l[0]==char and 'Error' not in l.split()[1]]))
                f.write('\n')
        if source_dir!=outdir:
            shutil.copy(file.replace('.ann','.txt'), file.replace(source_dir,outdir).replace('.ann','.txt'))
        for type,eid,ori_line in removed:
            changes.append({
                'id':file.split('/')[-1],
                'error type':type,
                'brat id':eid,
                'brat_ori_sent':ori_line
            })
    df=pd.DataFrame(changes)
    df.to_csv(SUMMARY_FILE,index=False)