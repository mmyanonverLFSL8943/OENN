# OENN
Anonymous code submission for ICML. 

## A brief guide to using OENN for Ordinal embedding.

### To run a baseline experiment:
```python
python train.py -d DATASET_NAME -bs BATCH_SIZE -lr LEARNING_RATE -ep NUM_EPOCHS -l LAYERS -hl HIDDEN_LAYER_WIDTH -minD MIN_EMBEDDING_DIM -maxD MAX_EMBEDDING_DIM -tr TRIPLET_CONSTANT
```

### To generate embeddings on new items:
```python
python test.py -mp TRAIN_MODEL_PATH -d DATASET_NAME -bs BATCH_SIZE -lr LEARNING_RATE -ep EPOCHS -l LAYERS -hl HIDDEN_LAYER_WIDTH -dim EMBEDDING_DIMENSION -tr NUM_TEST_TRIPLETS 
```

### To generate reconstruction of 2D datasets:
```python
python reprojection_of_2d_datasets.py -d DATASET_NAME -bs BATCH_SIZE -lr LEARNING_RATE -ep EPOCHS -l LAYERS
```

### To run a hyperparameter search experiment:
```python
python hyperparams/run_experiments_architecture_search.py -d DATASET_NAME -bs BATCH_SIZE -lr LEARNING_RATE -ep EPOCHS -min_n MIN_N -max_n MAX_N -min_hl MIN_HL_WIDTH -max_hl MAX_HL_WIDTH -minD MIN_EMBEDDING_DIM -maxD MAX_EMBEDDING_DIM -reps NUM_REPS
```

### To visualize the embedding output using Tsne:
```python
python vis_utils/tsne_cuda.py -mp MODEL_PATH -data DATASET_NAME -hl NUM_HL_UNITS -dim EMBEDDING_DIM -l LAYERS -p PERPLEXITY
```

**Arguments:**

-**DATASET_NAME:** Name of the dataset. (For a list of available dataset names, see below.)\
-**BATCH_SIZE:** Batch size used for training.\
-**LEARNING_RATE:** Learning rate.\
-**NUM_EPOCHS:** Number of epochs to train.\
-**LAYERS:** Number of layers of OENN.\
-**HIDDEN_LAYER_WIDTH:** Scaling factor to determine the number of units in each hidden layer = (120 + (hl * dim * log2(n))).\
-**MIN_EMBEDDING_DIM:** Minimum embedding dimension.\
-**MAX_EMBEDDING_DIM:** Maximum embedding dimension.\
-**TRIPLET_CONSTANT:** Scaling factor that determines the number of triplets generated = (tr * n * dim * log2(n)).\
-**TRAIN_MODEL_PATH:** Path to OENN trained on the train set.\
-**EMBEDDING_DIM:** Embedding dimension.\
-**NUM_TEST_TRIPLETS:** Number of new triplets to be generated between test and the training sets.\
-**MODEL_PATH:** Path to a saved model/checkpoint.\
-**MIN_N:** Minimum number of items to run the experiment over.\
-**MAX_N:** Maximum number of items to run the experiment over.\
-**NUM_REPS** Number of repetitions to run the experiment over. \
-**NUM_HL_UNITS:** Number of units in each hidden layer. \
-**PERPLEXITY:** Perplexity parameter for T-SNE visualization.

## Datasets
**General usage:** 
```python
data, labels = data_select_utils.select_dataset(dataset_name)
```

**To subsample:**
```python
data, labels = data_select_utils.select_dataset(dataset_name, subsample=True, n=10000)
```

### Real-world datasets
1. MNIST (*mnist*)
2. Fashion MNIST (*fmnist*)
3. EMNIST (*emnist*)
4. KMNIST (*kmnist*)
5. USPS (*usps*)
6. Newsgroup 20 (*news*)
7. Forest covertype (*cover_type*)
8. CHAR (*char*)
### List of 2D datasets.
1. Aggregation (*aggregation*)
2. Compound (*compound*)
3. D31 (*d31*)
4. Flame (*flame*)
5. Path (*path_based*)
6. R15 (*r15*)
7. Birch1 (*birch1*)
8. Birch2 (*birch2*)
9. Birch3 (*birch3*)
10. Spiral (*spiral*)
11. Worms (*worms*)
12. T48K (*t48k*)
13. Two moons (*moons*)
14. Circles (*circles*)
15. Blobs (*blobs*)
### Datasets with variable dimension.
**Usage:** 
```python
data, labels = data_select_utils.select_dataset(dataset_name='uniform', input_dim=2, subsample=True, n=10000)
```
1. Mixture of Gaussians (*gmm*)
2. Uniform distribution (*uniform*)
3. MNIST projected onto its principal components (*mnist_pc*)
4. USPS projected onto its principal components (*usps_pc*)
5. CHAR projected onto its principal components (*char_pc*)

## Examples for running experiments:

* for running an baseline experiment:
```python
python train.py -d mnist -bs 10000 -lr 1e-03 -ep 500 -l 3 -hl 1 -minD 5 -minD 5
```
* switch to a different dataset
```python
python train.py -d fmnist -bs 10000 -lr 1e-03 -ep 50 -l 3 -hl 1 -minD 5 -minD 5
```
* loop over dimensions
```python
python train.py -d mnist -bs 10000 -lr 1e-03 -ep 50 -l 3 -hl 1 -minD 5 -minD 10
```
* for hyperparams search
```python
python hyperparams/run_experiments_architecture_search.py -d mnist -bs 10000 -lr 1e-3 -ep 5 -min_n 3 -max_n 7 -min_hl 1 -max_hl 2 -minD 5 -maxD 10
```

* for 2D reconstructions
```python
python reprojection_of_2d_datasets.py -d r15 -bs 10000 -lr 1e-03 -ep 2 -l 3
```

