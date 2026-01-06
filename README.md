# MergeDNA
This is the use of the ToMe algorithm (adaptive DNA tokenization thru merging ) introduced in [MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](https://arxiv.org/abs/2511.14806) to train a toy model with a masked-prediction objective. 

The paper introduces a dna sequence tokenisation scheme that merges similar adjacent tokens. This enables the implicit categorisation of important vs repetitive regions. This merging is the outcome of a simple score outputing neural network trained along with task encoders. Therefore, this model is effectively using a hierarchical autoencoder architecture. 

## Architecture

The model uses a hierarchical autoencoder architecture:

```
Input DNA Sequence
       ↓
┌─────────────────┐
│  Local Encoder  │  ← Embeds bases + Token merging (N → L tokens)
└─────────────────┘
       ↓
┌─────────────────┐
│ Latent Encoder  │  ← Full-attention transformer for global context
└─────────────────┘
       ↓
┌─────────────────┐
│ Latent Decoder  │  ← Decodes from latent space
└─────────────────┘
       ↓
┌─────────────────┐
│  Local Decoder  │  ← Unmerges tokens (L → N)
└─────────────────┘
       ↓
Reconstructed Sequence
```
MergeDNA, thus learns to 'compress' DNA sequences by merging similar adjacent tokens dynamically. This ends up performing better/comparable than other standard tokenisation methods like bpe, state space model approaches, VQDNA (another dynamic tokenizer):
- **Adaptive tokenization** - Learns which regions are important vs. repetitive
- **Efficient representation** - Reduces sequence length while preserving biological information
- **Context-aware modeling** - Uses both local and global context for representation learning

To obtain these benefits, the model is trained with three unsupervised objectives:

| Loss | Description |
|------|-------------|
| **MTR** | Merged Token Reconstruction (reconstruct original DNA from merged tokens; cross entropy loss) |
| **Latent MTR** | forces disentangles representation learning as well |
| **AMTM** | Adaptive Masked Token Modeling (gives a higher masking probability on high-information regions or tokens not easily merged) |


## Next Steps, summary of the implementation: 
1. I've added a minimally complete implementation that complete the core forward training of the paper, i.e. the merging and masked learning. More task specific decoders can be added in the backbone.py and local_modules.py files like the promoter classification would need an classification decoder instead of the current version. 
2. The dataset originally downloaded for the promoter classification task was repurposed for the unsupervised training objective. Since the class sizes of the promoter and non-promoter sequences were similar - this ideally should not create a huge problem for a toy model. However the results seem to suggest: (a) it was a fairly simple task, (b) the data had not much variability. 
3. To scale this from a toy version to larger genomic sequences, the configs needs to be updated, along with more sound architectural decisions like encoder structure. 

quick summary (wrote after reading the paper):
Problem - genomic sequences like DNA are huge (> millions of base pairs), it is difficult to define a vocabulary over these sequences to train a machine learning model over. 
Solution - dynamic tokenization as in train the model to skim over repretivive sequences and focus on information dense parts in the sequence. 

The method to train this model is introduced in the paper, through the following steps: 
1. ToMe - a model that goes over the sequences and if they are repetitive, moves them into a single token and leave the other parts as is. 
2. Latent encoder - the model transforms the tokenized sequences into a latent space that is then trained using an appropriate objective. 

Training tasks: 
- re constructing the ToMe breakdown into the original sequence 
- masked training over the encoded embeddings over the dynamic tokens 

# some outcomes I've noticed 
The reconstruction task is fairly easy, therefore the paper also points to >98% of f1 scores. The latest training run over 10 epochs shows a reasonably high accuracy due to this as well. An additional plot is added in assets - the plot shows a close-up view of all the losses. All training was done on CPU. 

Classification as well as other benchmarks used in the paper also are not as challenging to learn, therefore the real value of using this approach for tokenization should come in based off the downstream task of zero-shot generation, or a use-case of genomic sequence embeddings. 

## Repository Structure

```
improbable-adventure/
├── mergedna/                    
│   ├── __init__.py              
│   ├── backbone.py              # MergeDNAModel, LatentEncoder, LatentDecoder
│   ├── local_modules.py         # LocalEncoder, LocalDecoder
│   ├── merging.py               # Bipartite matching, token merging
│   ├── dataloader.py            # DNADataset, data loading utilities
│   └── utils.py                 # Transformer blocks, positional embeddings
│
├── notebooks/                   
│   ├── _01_promoter.ipynb       # needs further work. 
│   └── _02_pretrain.ipynb       # Full pre-training pipeline
│
├── scripts/                     
│   ├── download_data.py         # Download genomic benchmark dataset used for toy model
│   ├── pretrain.py              
│   └── inference.py            
│
├── data/                        
│   └── human_nontata_promoters/ # Promoter classification dataset
│
├── checkpoints/                 
│
├── pyproject.toml               
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/improbable-adventure.git

# Install dependencies with uv
uv sync

# run the download_data file 

# run the _01_pretrain file. 
```

Available arguments:
- `--data_dir`: Path to dataset directory
- `--checkpoint_dir`: Where to save checkpoints (default: `checkpoints`)
- `--dim`: Model dimension (default: 64)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.001)
- `--merge_ratio`: Token merge ratio (default: 0.5)
- `--lambda_latent`: Weight for Latent MTR loss (default: 0.25)

### LocalEncoder & Decoder
Embeds DNA bases and performs token merging via bipartite soft matching. Reduces sequence length while preserving important information.
Unmerges tokens back to original length using the source map (ownership matrix) and applies refinement attention.

### LatentEncoder & Decoder
Stack of transformer blocks with a GlobalTokenSelector for adaptive token selection in the Latent MTR path.
Decodes from latent space back to the local token representation.
