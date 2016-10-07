import os
import pandas as pd 

def main():
    os.chdir('../DO_NOT_UPLOAD_THIS_DATA/0-output/')
    df1 = pd.read_csv("total_corpus_smry.csv")
    minv = df1['id'].max() + 1
    df2 = pd.read_csv("2012_aurora_shooting_first_sentence_numtext.csv")
    maxv = df2.shape[0]
    df3 = pd.DataFrame(pd.Series(range(minv, minv+maxv)))
    df3.columns = ['setence_id']
    df3.to_csv('2012_aurora_sentence_numtext.csv', index=False)

    print("Finished")

if __name__ == '__main__':
    main()