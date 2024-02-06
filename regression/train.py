import os, argparse, random
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
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

print('mode: regression')
print('num_estimators: ', args.num_estimators)
print('')
print('Training random forest regression model...')

clf = RandomForestRegressor(n_estimators = args.num_estimators)
clf.fit(X_train, Y_train)

print('Training random forest regression model OK')

print('')
print('Calculating random forest accuracy...')

predicted_y = clf.predict(X_test)
print('r^2: ' + str(metrics.r2_score(Y_test, predicted_y)))
print('mse: ' + str(metrics.mean_squared_error(Y_test, predicted_y)))
print('log(mse): ' + str(metrics.mean_squared_log_error(Y_test, predicted_y)))

print('')
print("Saving model...")
with open(os.path.join(args.out_directory, 'model.pkl'),'wb') as f:
    pickle.dump(clf,f)
print("Saving model OK")
