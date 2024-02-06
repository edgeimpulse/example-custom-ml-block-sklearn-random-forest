import os, argparse, random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Train a random forest model')
parser.add_argument('--data-directory', type=str, required=True,
                    help='Where to read the data from')
parser.add_argument('--out-directory', type=str, required=True,
                    help='Where to write the data')
parser.add_argument('--num-estimators', type=int, required=False, default=10,
                    help='The number of trees in the forest.')

args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.makedirs(args.out_directory, exist_ok=True)

X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'))
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'))
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

num_classes = Y_train.shape[1]

Y_train = np.argmax(Y_train, axis=1)
Y_test = np.argmax(Y_test, axis=1)

print('num classes: ' + str(num_classes))
print('mode: classification')
print('num_estimators: ', args.num_estimators)
print('')
print('Training random forest classifier...')

clf = RandomForestClassifier(n_estimators = args.num_estimators)

clf.fit(X_train, Y_train)

print(' ')
print('Calculating random forest accuracy...')

num_correct = 0
for idx in range(len(Y_test)):
    pred = clf.predict(X_test[idx].reshape(1, -1))
    if num_classes == 2:
        if Y_test[idx] == int(round(pred[0])):
            num_correct += 1
    else:
        if Y_test[idx] == pred[0]:
            num_correct += 1
print(f'Accuracy (validation set): {num_correct / len(Y_test)}')

print('')
print("Saving model...")
with open(os.path.join(args.out_directory, 'model.pkl'),'wb') as f:
    pickle.dump(clf,f)
print("Saving model OK")
