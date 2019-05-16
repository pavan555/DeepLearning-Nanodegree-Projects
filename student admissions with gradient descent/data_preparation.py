import numpy as np
import pandas as pd

admissions = pd.read_csv('../datasets/student_data.csv')

# Make dummy variables for rank
#applying one hot encoding for rank as it is categorized into rank1,rank2,rank3,rank4
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)

#dropping the rank column because there is no use for us
data = data.drop('rank', axis=1)

# Standarize features

for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
#splliting the given data into Output(admit column) and input (ranks,gre,gpa)
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']