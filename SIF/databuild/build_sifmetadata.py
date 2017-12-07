import os
import itertools
import pandas as pd

def summarizedata(filename):
    df = pd.read_csv(filename)

    sumry_tokens = df['summary'][0].split(" ")
    sumry_len = len(sumry_tokens)
    sumry_ntokens = len(set(sumry_tokens))

    sent_tokens = list(itertools.chain(*[x for x in df['sentence'].str.split(" ")]))
    sent_len = len(sent_tokens)
    sent_ntokens = len(set(sent_tokens))
    sent_nchar = len(' '.join(sent_tokens))

    sdic = {
        'filename': filename,
        'summary_len': sumry_len,
        'summary_ntokens': sumry_ntokens,
        'nsentences': df.shape[0],
        'sentences_len': sent_len,
        'sentences_ntokens': sent_ntokens,
        'sentences_nchar': sent_nchar
    }

    return pd.DataFrame.from_dict(sdic, orient='index').T

def main():
    inputpath = "/home/francisco/GitHub/DQN-Event-Summarization/data/sif/"
    outputpath = "/home/francisco/GitHub/DQN-Event-Summarization/SIF/data/metadata/"

    outfile = "cnn_dm_metadata.csv"
    fulloutfile = os.path.join(outputpath, outfile)

    files = os.listdir(inputpath)

    for i, file_i in enumerate(files):
        output = summarizedata(os.path.join(inputpath, file_i))
        if i == 0:
            metadata = output
        else:
            metadata = pd.concat([metadata, output], axis=0)

    metadata.to_csv(fulloutfile, index=False)
    print('Metadata created to %s' % fulloutfile)

if __name__ == "__main__":
    main()
