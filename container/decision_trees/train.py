import os
import json
import sys
# import subprocess
import traceback
from datetime import datetime, timezone
from collections import OrderedDict
import yaml
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn import tree
import pickle

# Sagemaker requirements for interfacing with S3 buckets
# during any training job
"""
Script to run a training job in Sagemaker
TODO: Replace yaml load with configparser
"""


with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
prefix = config["dir_root"]  # TODO: Change this to /opt/ml
# These are in alignment with Sagemaker's guidelines
input_dir = os.path.join(prefix, config["input_dir_prefix"])
train_input_dir = os.path.join(input_dir, config["train_channel_name"])
config_dir = os.path.join(prefix, config["config_dir_prefix"])
output_dir = os.path.join(prefix, config["output_dir_prefix"])
model_dir = os.path.join(prefix, config["model_dir_prefix"])
model_name = config["model_name"]
param_path = os.path.join(config_dir, config["hyperparam_file_name"])
default_param_path = os.path.join("", config["default_hyperparam_file_name"])
failure_path = os.path.join(output_dir, "failure")

print("Files used in training: {}".format(os.listdir(train_input_dir)))

def train():
    print('Starting the training.')
    try:
        # Read in any hyperparameters that the user passed with the training job
        with open(param_path, 'r') as tc:
            trainingParams = json.load(tc)

        # Take the set of files and read them all into a single pandas dataframe
        input_files = [ os.path.join(train_input_dir, file) for file in os.listdir(train_input_dir) ]
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(train_input_dir, config["train_channel_name"]))
        raw_data = [pd.read_csv(file, header=None) for file in input_files ]
        train_data = pd.concat(raw_data)

        # labels are in the first column
        train_y = train_data.iloc[:,0]
        train_X = train_data.iloc[:,1:]

        # Here we only support a single hyperparameter. Note that hyperparameters are always passed in as
        # strings, so we need to do any necessary conversions.
        max_leaf_nodes = trainingParams.get('max_leaf_nodes', None)
        if max_leaf_nodes is not None:
            max_leaf_nodes = int(max_leaf_nodes)

        # Now use scikit-learn's decision tree classifier to train the model.
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        clf = clf.fit(train_X, train_y)

        # save the model
        model_path = os.path.join(model_dir, model_name)
        print("Model Path {}".format(model_path))
        with open(model_path, 'wb') as out:
            pickle.dump(clf, out)

        training_metadata = {
            'arch': 'decision_tree',
            'model_path': model_path,
            'num_classes': train_y.nunique(),
            'training_status': 'Success'
        }
        with open(os.path.join(model_dir, 'training_metadata.yaml'), 'w') as f:
            yaml.dump(training_metadata, f)
        print('Training complete.')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_dir, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == "__main__":
    train()
