import argparse
from cgitb import text
from genericpath import isfile
from html import entities
from pickletools import read_unicodestring1
import spacy
import json
import os
import logging
import glob
from tqdm import tqdm

#import the dataloader for relationships
from brat_scoring.brat import *
from brat_scoring.labels import *
from brat_scoring.document import *
from config import *

def load_json(filepath="config.json"):
    f = open(filepath)
    return json.load(f)

def dump_json(dic,outfile):
    with open(outfile, 'w') as openfile:
        json.dump(dic,openfile,indent=4)
    return

def create_sentence(tokens,entities,relations,events,doc_id,sent_id,config,sent_offset,sent_token_offsets):
    entity_ls=[]
    #entities
    entities=[entity for entity in entities if entity.type_ not in ["Form","Route"] and "rror?" not in entity.type_]
    #sent_offset=min([entity.token_start for entity in entities])
    for entity in entities:
        if  True:
            entence_dic={
                            "type": entity.type_, 
                            "subtype":entity.subtype if entity.subtype else "",
                            "start": entity.token_start-sent_offset, 
                            "end": entity.token_end-sent_offset,
                            "text":" ".join(tokens[entity.token_start-sent_offset:entity.token_end-sent_offset])
                            }
            assert entity.token_start<entity.token_end, "BRAT annotation error in doc {} span {}".format(doc_id,entity.tokens) 
                
                #check if the span matches the document
            if False:#entity.tokens!=tokens[entity.token_start-sent_offset:entity.token_end-sent_offset]:
                    logging.warning("token mismatch in doc {} span {}".format(doc_id,entity.tokens))
                    return None
            else:
                entity_ls.append(entence_dic)
    entity_ls = sorted(entity_ls, key=lambda d: (d["start"],d["end"])) 
    #relations
    relation_ls=[]
    for relation in relations:
        if relation.entity_a.arguments[0] in entities and relation.entity_b.arguments[0] in entities:
            head_index=[i for i in range(len(entity_ls)) \
                        if entity_ls[i]["start"]+sent_offset==relation.entity_a.arguments[0].token_start \
                        and entity_ls[i]["end"]+sent_offset==relation.entity_a.arguments[0].token_end \
                        and entity_ls[i]["type"]==relation.entity_a.arguments[0].type_ ]
            
            tail_index=[i for i in range(len(entity_ls)) \
                        if entity_ls[i]["start"]+sent_offset==relation.entity_b.arguments[0].token_start \
                        and entity_ls[i]["end"]+sent_offset==relation.entity_b.arguments[0].token_end \
                        and entity_ls[i]["type"]==relation.entity_b.arguments[0].type_ ]
            
            if len(head_index)==1 and len(tail_index)==1:
                #print(relation.entity_a.arguments[0])
                relation_dic={
                            "type": relation.role, 
                            "head": head_index[0], 
                            "tail": tail_index[0],
                            "head_text":entity_ls[head_index[0]]['text'],
                            "tail_text":entity_ls[tail_index[0]]['text'],
                            }
                relation_ls.append(relation_dic)
    relation_ls = sorted(relation_ls, key=lambda d: (d["head"],d["tail"])) 
    
    #events as relations
    for event in events:
        head=event.arguments[0]
        for tail in event.arguments[1:]:
            if tail in entities: #only consider within sentence relations
                
                head_index=[i for i in range(len(entity_ls)) \
                        if entity_ls[i]["start"]+sent_offset==head.token_start \
                        and entity_ls[i]["end"]+sent_offset==head.token_end \
                        and entity_ls[i]["type"]==head.type_ ]
            
                tail_index=[i for i in range(len(entity_ls)) \
                            if entity_ls[i]["start"]+sent_offset==tail.token_start \
                            and entity_ls[i]["end"]+sent_offset==tail.token_end \
                            and entity_ls[i]["type"]==tail.type_ ]
                if len(head_index)==1 and len(tail_index)==1:
                    relation_dic={"type": event.arguments[0].type_+"-"+tail.type_, 
                                    "head": head_index[0], 
                                    "tail": tail_index[0]}
                    relation_ls.append(relation_dic)
            else:
                logging.warning("cross-sentence relationship at doc ({}) sentence i({}) with head ({}) and tail ({}) "\
                    .format(doc_id,sent_id,event.arguments[0].tokens,sent_id,tail.tokens))
    #entity_ls=[e for e in entity_ls if e["type"] not in ["Form","Route"] and "rror?" not in e["type"]]
    #relation_ls=[]#[r for r in relation_ls if "Drug"  in r["type"]]
    
    return {
            config["ID"]:doc_id,
            config["SENTENCE_ID"]:sent_id,
            "offset":sent_offset,
            config["TOKEN"]:tokens,
            'token_offsets':sent_token_offsets,
            config["ENTITY"]:entity_ls,
            config["RELATION"]:relation_ls,
            }

#record the types in the dataset
def write_types(entities,events,relations,type_dic=None):
    if not type_dic:
        type_dic={"entities":{},"relations":{},'tokens':{}}
    
    for entity in entities:
        if entity.type_ not in type_dic["entities"]:
            type_dic["entities"][entity.type_]= {"short": entity.type_,
                                    "verbose": entity.type_}
    for event in events:
        for tail in event.arguments[1:]:
            relation_type=event.arguments[0].type_+"-"+tail.type_
            if relation_type not in type_dic["relations"]:
                type_dic["relations"][relation_type]= {"short": relation_type,
                                        "verbose": relation_type,
                                        "symmetric": False}
    for relation in relations:
            relation_type=relation.role
            type_dic["relations"][relation_type]= {"short": relation_type,
                                        "verbose": relation_type,
                                        "symmetric": False}

    return type_dic
    
def doc2json(doc,config,dir,type_dic):
    doc_ls=[]
    #converting 2d lists into 1d, and to get the values
    tokens= [t for sent in doc.tokens for t in sent]
    offsets= [t for sent in doc.token_offsets for t in sent]
    
    #loading entities and relations
    entities=doc.entities()
    relations=doc.event_relations() #not used yet
    events=doc.events() #events between triggers
    #record the entity types and relations for the Spert input
    type_dic=write_types(entities,events,relations,type_dic)
        
    #sentence_start
    L=len(offsets)
    #print(offsets)
    
    sent_start=[sum([len(k) for k in doc.token_offsets[:i]]) for i,l in enumerate(doc.token_offsets)]#[ i for i in range(L) if (i>0 and i<L-1 and tokens[i-1] in config["SENTENCE_SEP"]) or i==0]
    #print(len(sent_start))
    #check if entity aligns with the document
    for entity in entities:
        assert tokens[entity.token_start:entity.token_end]==entity.tokens, \
            "entity not match text {}, {}, {}, {}".format(entity.type_,entity.tokens,entity.token_start,entity.token_end,entity.sent_index)
    
    #create the dic for each sentence:
    for i in range(len(sent_start)):
        idx_start,idx_end=sent_start[i], sent_start[i+1] if i<len(sent_start)-1 else len(tokens)
        if idx_start+config["MAXLEN"]>=idx_end:
            #limit the lenth of input sentence
            sent_tokens=tokens[idx_start:idx_end]
            sent_entities=[entity for entity in entities if entity.token_start>=idx_start and entity.token_end<=idx_end]
            sent_relations=[relation for relation in relations if relation.entity_a.arguments[0] in sent_entities and relation.entity_b.arguments[0] in sent_entities]
            sent_events=[relation for relation in events if relation.arguments[0] in sent_entities]
            sent_token_offsets=[offsets[i] for i in range(idx_start,idx_end)]
            sent_dic=create_sentence(sent_tokens,sent_entities,sent_relations,sent_events,doc.id,i,config,idx_start,sent_token_offsets)
            if sent_dic:
                doc_ls.append(sent_dic)
    return doc_ls,type_dic

def main(dir,outname=''):
    #loading config file
    config=spert_configs
    type_dic=None
    
    #logging 
    logging.basicConfig(filename=dir.replace("*/","_")+"_log.log",# encoding='utf-8',
                        format='(%(levelname)s)\t%(message)s',
                        filemode='w',
                        level=logging.INFO) #can set it to debug
 
    #loading tokenizer
    tokenizer = spacy.load(config["SPACY_MODEL"])
    
    dataset=[]
    
    #transfering every file in the direction to the json file
    filenames=glob.glob(dir+"/*.txt")
    filenames=[text_file for text_file in filenames if not ( "_ori" in text_file or "temp" in text_file)]
    #pbar = tqdm(total=len(filenames), desc='File reformatting')
    for text_file in tqdm(filenames):
        print(text_file)
        #check if annotation exists
        ann_file=text_file.replace(".txt",".ann")
        if not os.path.isfile(ann_file):
            logging.warning("file {} does not have annotations!".format(text_file))
            continue
        
        #extract the information
        with open(ann_file, 'r', encoding=ENCODING) as f:
            ann = f.read()
        with open(text_file, 'r', encoding=ENCODING) as f:
            text = f.read()
        id = os.path.splitext(os.path.relpath(text_file, dir))[0]
        
        doc = Document( \
                id = id,
                text = text,
                ann = ann,
                tags = None,
                tokenizer = tokenizer,
                strict_import = False
                )
        
        doc_ls,type_dic=doc2json(doc,config,dir,type_dic)
        if doc_ls:
            dataset.extend(doc_ls)
        logging.info("file {} included in json".format(text_file))
        #pbar.update(1)
    if not outname:
        dump_json(dataset,dir.replace("*/","_")+"_spert.json")
        dump_json(type_dic,dir.replace("*/","_")+"_types.json")
    else:
        dump_json(dataset,outname+"_spert.json")
        dump_json(type_dic,outname+"_types.json")        
    return

if __name__ == "__main__":
    for split in ['train','valid','test']: 
        main(f'dataset/BRAT/{split}',f'dataset/spert/{split}')