"""
ThermoPalm: Predict melting temperature of wild type proteins with a protein language embedding
"""




import numpy as np
import pandas as pd

import torch

import joblib
import os
import argparse
import sys
sys.path.insert(1, './')

import warnings
warnings.filterwarnings('ignore')
from builtins import print as builtin_print








def parse_arguments():
    '''Parse command-line training arguments'''
    
    parser = argparse.ArgumentParser(description="Predict Tm of proteins")
    parser.add_argument('--fasta_path', type=str, 
                        help='Path to fasta file of protein sequences')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='Directory to which prediction results will be written')
    parser.add_argument('--csv_name', type=str, default='prediction.csv', 
                        help='Name of csv file to which prediction results will be written')
    parser.add_argument('--verbose', type=int, default=1, 
                        help="If 1, print out progress")
    args = parser.parse_args()

    return args






def print(*args, **kwargs):
    
    kwargs.setdefault('flush', True)
    return builtin_print(*args, **kwargs)






class ProtT5Model():
    
    def __init__(self, modelname, device='cuda'):

        self.t5_names = [
            'prot_t5_xl_uniref50', 
            'prot_t5_xxl_uniref50', 
            'prot_t5_xl_bfd', 
            'prot_t5_xxl_bfd'
            ]
        self.modelname = modelname
        assert self.modelname in self.t5_names
        self.device = device
        self.feature_axis = 0
        self.load()
        
    
    
    def load(self):

        from transformers import T5EncoderModel, T5Tokenizer
        name = f"Rostlab/{self.modelname}"
        if 't5' in self.modelname:
            self.model = T5EncoderModel.from_pretrained(name)
            self.tokenizer = T5Tokenizer.from_pretrained(name, do_lower_case=False)
        
        self.layers = len(self.model._modules['encoder']._modules['block']._modules)
        self.model.eval()
        
        
    def encode(self, sequence):

        torch.cuda.empty_cache()
        for char in 'UZOB':
            sequence = sequence.replace(char, 'X')
        sequence = ' '.join(sequence)
        ids = self.tokenizer.batch_encode_plus([sequence], 
                                               add_special_tokens=True, 
                                               padding=True)
        model = self.model.to(self.device)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        with torch.no_grad():
            emb = model(input_ids=input_ids,attention_mask=attention_mask)
            emb = emb.last_hidden_state.cpu()[0]
        
        return emb    






def seq_embedder(model, seqdict, args):

    size = len(seqdict.keys())
    embs = {}

    if args.verbose:
        print("Retrieving embeddings for {size} proteins with Prot-T5-XL-U50")

    for i,(acc, seq) in enumerate(seqdict.items()):
        
        if args.verbose:
            print(f"{i+1} of {size}")
        
        x = model.encode(seq)
        x = x.numpy().mean(axis=model.feature_axis) # Average embeddings
        embs[acc] = x
        if i == size - 1:
            df = pd.DataFrame(embs).transpose()   
    
    return df






def load_ridge_model():
    '''Return trained ridge regression model and statistics for feature scaling'''
    
    this_dir, this_filename = os.path.split(__file__)
    models_path = os.path.join(this_dir, 'ridge_model.pkl')
    stats_path = os.path.join(this_dir, 'scaler_stats.csv')
    model = joblib.load(models_path)
    stats = pd.read_csv(stats_path, index_col=0)
    means = stats.iloc[:,0].values.reshape(1,-1)
    stds = stats.iloc[:,1].values.reshape(1,-1)
    
    return model, means, stds
      
    




def read_fasta(fasta, return_as_dict=False):
    '''Read the protein sequences in a fasta file. If return_as_dict, return a dictionary
    with headers as keys and sequences as values, else return a tuple, 
    (list_of_headers, list_of_sequences)'''
    
    headers, sequences = [], []

    with open(fasta, 'r') as fast:
        
        for line in fast:
            if line.startswith('>'):
                head = line.replace('>','').strip()
                headers.append(head)
                sequences.append('')
            else :
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq

    if return_as_dict:
        return dict(zip(headers, sequences))
    else:
        return (headers, sequences)






def main():

    args = parse_arguments()
    
    # Prepare arguments and paths
    if args.verbose:
        print("Starting job")
        print("Parsing arguments")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.csvfile = f"{args.save_dir}/{args.csv_name}"
    if args.verbose:
        print(f"Predictions will be written to {args.csvfile}")
    
    # Load PLM
    if args.verbose:
        print("Loading protein language model")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ProtT5Model('prot_t5_xl_uniref50', device=device) 
    
    # Read sequences
    heads, seqs = read_fasta(args.fasta_path)
    accs = [head.split()[0] for head in heads]
    heads, seqs, accs = [np.array(item) for item in (heads, seqs, accs)]
    seqdict = dict(zip(accs, seqs))
    if args.verbose:
        print(f"Read {len(heads)} sequences from {args.fasta_path}")

    # Embed sequences
    dfemb = seq_embedder(model, seqdict, args)
    assert dfemb.shape[1] == 1024
    
    
    # Load ridge model
    model, means, stds = load_ridge_model()
    
    
    # Standardize and predict
    if args.verbose:
        print("Predicting Tm")
    X = (dfemb.values - means) / (stds + 1e-8)
    ypred = model.predict(X)
    dfy = pd.Series(ypred, index=dfemb.index)
    dfy.to_csv(args.csvfile)
    if args.verbose:
        print("Done!")
    
    
   
        


if __name__ == '__main__':

    main()
    
    
    
        
        





