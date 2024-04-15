from glob import glob
import shutil
import os

def print_relation(anns,file):
    duplicates=[]
    relations=[]
    source_relations=[]
    for l in anns:
        if l[0]=='R':
            relations.append(l.split()[2:])
            source_relations.append(l)
    for r in relations:
        if relations.count(r)>1:
            targets=[source_relations[idx].split()[0] for idx,r2 in enumerate(relations) if r2==r]
            duplicates.append('&'.join(targets))
            anns.append(f'R{len(duplicates)+1000}\tRelationError {r[0]} {r[1]}')

    return list(set(duplicates)),anns


if __name__ == "__main__":

    # source_dir=f"" # the source directory containing all ann files.
    # outdir="" # the output directory to store all the BRAT files with errors.
    # files=glob(f"{source_dir}/*.ann")
    # SUMMARY_FILE="" # the output csv file for a summary of errors.

    # pring the files with double relation
    split='test'
    source_dir=f"/home/velvinfu/data/ProstateDLBCLPostDiagnosis/R15_dataset_paper/final3/{split}_cleaned" # the source directory containing all ann files.
    outdir=f"/home/velvinfu/data/ProstateDLBCLPostDiagnosis/R15_dataset_paper/final3/{split}_relation" # the output directory to store all the BRAT files with errors.
    
    SUMMARY_FILE=f"/home/velvinfu/data/ProstateDLBCLPostDiagnosis/R15_dataset_paper/final3/{split}_relation.csv" # the output csv file for a summary of errors.

    files=glob(f"{source_dir}/*.ann")

    os.makedirs(outdir,exist_ok=True)
    changes=[]
    for file in files:
        anns=[l.strip() for l in open(file).read().split('\n') if l.strip()]
        duplicates,anns=print_relation(anns,file)
        if duplicates:
            with open(file.replace(source_dir,outdir),'w') as f:
                f.write('\n'.join(anns))
            
            shutil.copy(file.replace('.ann','.txt'), file.replace(source_dir,outdir).replace('.ann','.txt'))
            for pair in duplicates:
                changes.append(f'{file.split("/")[-1]},{pair}')

    with open(SUMMARY_FILE,'w') as f:
        f.write('\n'.join(changes))