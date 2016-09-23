import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

os.chdir('/Users/franciscojavierarceo/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/corpus-data/')
os.listdir('./')

df1 = pd.read_csv('./2012_aurora_shooting.tsv.gz', sep='\t')
df2 = pd.read_csv('./2012_pakistan_garment_factory_fires.tsv.gz', sep='\t')
df3 = pd.read_csv('./hurricane_sandy.tsv.gz', sep='\t')
df4 = pd.read_csv('./wisconsin_sikh_temple_shooting.tsv.gz', sep='\t')

print( df1[df1['sentence'] < 1].shape, 
       df2[df2['sentence'] < 1].shape, 
       df3[df3['sentence'] < 1].shape, 
       df4[df4['sentence'] < 1].shape)

df1[df1['sentence'] < 1].to_csv("./2012_aurora_shooting_first_sentence.tsv", index=False, sep='\t') 
df2[df2['sentence'] < 1].to_csv("./2012_pakistan_garment_factory_fires_first_sentence.tsv", index=False, sep='\t') 
df3[df3['sentence'] < 1].to_csv('./hurricane_sandy_first_sentence.tsv', index=False, sep='\t') 
df4[df4['sentence'] < 1].to_csv('./wisconsin_sikh_temple_shooting_first_sentence.tsv', index=False, sep='\t') 

os.system('gzip ./2012_aurora_shooting_first_sentence.tsv')
os.system('gzip ./2012_pakistan_garment_factory_fires_first_sentence.tsv')
os.system('gzip ./hurricane_sandy_first_sentence.tsv')
os.system('gzip ./wisconsin_sikh_temple_shooting_first_sentence.tsv')

os.chdir('/Users/franciscojavierarceo/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/trec-2013-data/')

df5 = pd.read_csv('./nuggets.tsv.gz', sep='\t')
df5[df5['query_id']=='TS13.3'].to_csv("./aurora_nuggets.tsv", sep='\t')
df5[df5['query_id']=='TS13.2'].to_csv("./pakistan_nuggets.tsv", sep='\t')
df5[df5['query_id']=='TS13.6'].to_csv("./sandy_nuggets.tsv", sep='\t')
df5[df5['query_id']=='TS13.4'].to_csv("./wisconsin_nuggets.tsv", sep='\t')

os.system('gzip /Users/franciscojavierarceo/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/trec-2013-data/aurora_nuggets.tsv')
os.system('gzip /Users/franciscojavierarceo/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/trec-2013-data/pakistan_nuggets.tsv')
os.system('gzip /Users/franciscojavierarceo/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/trec-2013-data/sandy_nuggets.tsv')
os.system('gzip /Users/franciscojavierarceo/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/trec-2013-data/wisconsin_nuggets.tsv')