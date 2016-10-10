import os
import pandas as pd 

def build(input_file, output_file):
    os.chdir('../DO_NOT_UPLOAD_THIS_DATA/0-output/')
    df1 = pd.read_csv("total_corpus_smry.csv")
    minv = df1['id'].max() + 1
    df2 = pd.read_csv(input_file)
    maxv = df2.shape[0]
    df3 = pd.DataFrame(pd.Series(range(minv, minv+maxv)))
    df3.columns = ['setence_id']
    df3.to_csv( output_file, index=False)

    print("Finished")

if __name__ == '__main__':
  build('2012_aurora_shooting_first_sentence_numtext.csv',
        '2012_aurora_sentence_numtext.csv')

  build('2012_pakistan_garment_factory_fires_first_sentence_numtext.csv',
        '2012_pakistan_sentence_numtext.csv')

   build('hurricane_sandy_first_sentence_numtext.csv',
         'hurricane_sandy_sentence_numtext.csv')
