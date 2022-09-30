import tensorflow as tf
import deepchem as dc
from deepchem.utils.data_utils import load_from_disk
import train_HIV
from sklearn.model_selection import train_test_split

dataset_file= "HIV.csv"

dataset = load_from_disk(dataset_file)

featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(tasks=["HIV_active"], smiles_field="smiles", featurizer=featurizer)
dataset = loader.featurize(dataset_file)

Num_layer = 4
dim = [32,64,128,256]

#####  Data splitting into test:train = 2:8
X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.2)
X_train = X_train.astype(float)
X_test = X_test.astype(float)

model = train_HIV.train((X_train, X_test, y_train, y_test), Num_layer, dim)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=1)
print('model Accuracy:', test_acc)