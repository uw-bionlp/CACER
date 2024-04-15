
from glob import glob
import shutil
import os
# configurations for the BRAT file
ASSERTION="Assertion"
PROBLEM="Problem"

SPAN_WITH_ASSERTION=[ASSERTION,"Change","Severity"]
OK_TO_FLOAT=[]#["Possible_Drug","Drug"]


def read_ann(filename):
    anns=open(filename).read().split("\n")
    anns=[l for l in anns if l and not any([k in l for k in ["SpanFloating","NoVal","MultiVals"]])]

    # read tbs
    tbs={}
    for line in anns:
        if line and line[0]=="T":
            words=line.split("\t")
            tb_index,type,word_index,span=words[0],words[1].split(" ")[0]," ".join(words[1].split(" ")[1:]),"\t".join(words[2:])
            tbs[tb_index]=[type,word_index,span]
    
    # read entities
    entities={}
    for line in anns:
        if line and line[0]=="E":
            words=line.split()
            e_index=words[0]
            words=[w.split(":")[1] for w in words[1:]]
            entities[e_index]=words
    
    #read assertions
    assertions={}
    for line in anns:
        if line and line[0]=="A":
            words=line.split()
            assert len(words)==4
            assertions[words[2]]=assertions.get(words[2],[])+[words[0]]
    return tbs,entities,assertions,anns

def add_anns(out_csv,ls,label,anns,tbs,current_index):
    for tb in ls:
        if tb in tbs:
            anns.append(f"T{current_index}\t{label} {tbs[tb][1]}\t{tbs[tb][2]}")
            current_index+=1
    out_csv+=f"{len(ls)},"
    return anns,current_index,out_csv

def write_ann(anns,filename,source_dir,outdir):
    with open(filename.replace(source_dir,outdir),"w",encoding="utf-8") as f:
        f.write("\n".join(anns))
    txt_name=filename.replace(".ann",".txt")
    shutil.copy(txt_name,txt_name.replace(source_dir,outdir))
    return


if __name__ == "__main__":

    split='test'
    source_dir="" # the source directory containing all ann files.
    outdir="" # the output directory to store all the BRAT files with errors.
    SUMMARY_FILE=""
    
    split='train'
    source_dir=f"/home/velvinfu/data/ProstateDLBCLPostDiagnosis/R15_dataset_paper/final3/{split}_cleaned" # the source directory containing all ann files.
    outdir=f"/home/velvinfu/data/ProstateDLBCLPostDiagnosis/R15_dataset_paper/final3/{split}_entity" # the output directory to store all the BRAT files with errors.
    
    SUMMARY_FILE=f"/home/velvinfu/data/ProstateDLBCLPostDiagnosis/R15_dataset_paper/final3/{split}_entity.csv" # the output csv file for a summary of errors.

    files=glob(f"{source_dir}/*.ann")
    # the output csv file for a summary of errors.

    summary=[]
    os.makedirs(outdir,exist_ok=True)
    for filename in files:
        out_csv=filename.split("/")[-1]+","
        tbs,entities,assertions,anns=read_ann(filename)
        current_index=max([int(k[1:]) for k in tbs])+1

        #find floating tbs
        floating_tb=[]
        connect_tbs=[v for key in entities for v in entities[key]]
        for tb in tbs:
            if tb not in connect_tbs and tbs[tb][0] not in OK_TO_FLOAT:
                floating_tb.append(tb)
        anns,current_index,out_csv=add_anns(out_csv,floating_tb,"SpanFloating",anns,tbs,current_index)

        # spans without assertion
        assert_non=[]
        for tb in tbs:
            if tbs[tb][0] in SPAN_WITH_ASSERTION:
                if tb not in assertions:
                    assert_non.append(tb)
        anns,current_index,out_csv=add_anns(out_csv,assert_non,"NoVal",anns,tbs,current_index)

        # assertions spans with more than 1 values
        assert_multiple=[]
        for tb in assertions:
            if len(assertions[tb])>1:
                assert_multiple.append(tb)
        anns,current_index,out_csv=add_anns(out_csv,assert_multiple,"MultiVals",anns,tbs,current_index)



        # multiple spans with the exact same indices
        multi_label=[]
        ls=[[tb,tbs[tb][1]] for tb in tbs if tbs[tb][0]!=ASSERTION]
        t_tbs,t_index=[l[0] for l in ls], [l[1] for l in ls]
        for tb,index in ls:
            if t_index.count(t_index)>1:
                multi_label.append(tb)
        anns,current_index,out_csv=add_anns(out_csv,multi_label,"MultiLabel",anns,tbs,current_index)

        # problems without any assertion or multiple assertion
        no_assertion=[]
        multi_assertion=[]
        for all_tbs in entities.values():
            if all_tbs[0]in tbs and tbs[all_tbs[0]][0]==PROBLEM:
                assertion_count=len([tb for tb in all_tbs if tbs[tb][0]==ASSERTION])
                if assertion_count==0:
                    no_assertion.append(all_tbs[0])
                elif assertion_count>1:
                    multi_assertion.append(all_tbs[0])
        anns,current_index,out_csv=add_anns(out_csv,no_assertion,"NoAssert",anns,tbs,current_index)
        anns,current_index,out_csv=add_anns(out_csv,multi_assertion,"MultiAssert",anns,tbs,current_index)
        
        #output summary
        if floating_tb+assert_non+assert_multiple+multi_label+no_assertion+multi_assertion:
            write_ann(anns,filename,source_dir,outdir)
        summary.append(out_csv)

    with open(SUMMARY_FILE,"w") as f:
        f.write("filename,floating tbs,label without values, label with multiple values,multi labels for same span,Problem without assertion, Problem with multiple assertions\n")
        f.write("\n".join([l for l in summary if '0,0,0,0,0,0,' not in l]))

