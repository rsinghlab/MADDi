import io
import os
import numpy as np
import pandas as pd


def main():
    
    
    
    files = os.listdir("YOUR_PATH_TO_FILTERED_VCFS")
    diag = pd.read_csv("YOUR_PATH_TO_DIAGNOSIS_TABLE")[["index", "Group"]]
    
    vcfs = []
    
    for vcf_file in files:
        file_name = "YOUR_PATH_TO_FILTERED_VCFS" + vcf_file
        
        vcf = pd.read_pickle(file_name)
	
        vcf = vcf.drop(['#CHROM', 'POS', 'ID','REF','ALT','QUAL','FILTER','INFO', 'FORMAT'], axis=1)
        vcf = vcf.T
        vcf.reset_index(level=0, inplace=True)
        vcf["index"] = vcf["index"].str.replace("s", "S").str.replace("\n", "")
        merged = diag.merge(vcf, on = "index")
        merged = merged.rename(columns={"index": "subject"})
        d = {'0/0': 0, '0/1': 1, '1/0': 1,  '1/1': 2, "./.": 3}
        cols = list(set(merged.columns) - set(["subject", "Group"]))
        for col in cols:
            merged[col] = merged[col].str[:3].replace(d)
            idx = cols.index(col)
            if idx % 500 == 0:
                output_file = open('log_clean.txt','a')
                output_file.write("Percent done: " + str((idx/len(cols))*100) + "\n")
                output_file.close()
        
        merged.to_pickle(vcf_file + "clean.pkl")

        vcf = vcf.groupby('index', group_keys=False).apply(lambda x: x.loc[x.Group.idxmax()])

        vcfs.append(vcf)
    
    vcf = pd.concat(vcfs, ignore_index=True)
    vcf = vcf.drop_duplicates()
    vcf.to_pickle("all_vcfs.pkl")



    
if __name__ == '__main__':
    main()
    
