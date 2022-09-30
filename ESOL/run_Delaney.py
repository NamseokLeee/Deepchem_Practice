import tensorflow as tf
import deepchem as dc
from deepchem.utils.data_utils import load_from_disk
import train_Delaney


dataset_file= "delaney-processed.csv"

dataset = load_from_disk(dataset_file)

featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(tasks=["measured log solubility in mols per litre"], smiles_field="smiles", featurizer=featurizer)
dataset = loader.featurize(dataset_file)

Num_layer = 4
dim = [32,64,128,256]

#####  Data splitting into test:train = 2:8
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, frac_train= .8, frac_valid = 0, frac_test= .2)

##### Data Normalization (zero mean, unit variance)
transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]

for dataset in [train_dataset, test_dataset]:
    for transformer in transformers:
        dataset = transformer.transform(dataset)


model, history = train_Delaney.train((train_dataset,test_dataset), Num_layer, dim)

loss, mae, mse = model.evaluate(test_dataset.X, test_dataset.y, verbose=2)
print("best RMSE: %f\nbest r2value: %f" % (mse, 1-(mse/test_dataset.y.var())))