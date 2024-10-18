import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import itertools
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import  confusion_matrix, accuracy_score
from joblib import Parallel, delayed


class node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, depth=0, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth
        self.value = value #The value of the node if it is a leaf

    def is_leaf(self):
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(self, max_depth_limit=None, entropy_cutoff=None, splitting_criterion=None, min_samples_split_count=2, features=None):
        self.max_depth_limit = max_depth_limit
        self.entropy_cutoff = entropy_cutoff
        self.splitting_criterion = splitting_criterion
        self.min_samples_split_count = min_samples_split_count
        self.criterion_function = {
            'scaled_entropy': self._scaled_entropy,
            'gini': self._gini,
            'squared': self._squared_impurity,
        }.get(self.splitting_criterion)
        self.root_node = None
        self.leaf_node_count = 0
        self.depth = 0
        self.feature_list = features
        

    def fit(self, X, y):
        self.root_node = self._grow_tree(X, y)       


    def get_params(self, deep=True):
        return {
            'max_depth_limit': self.max_depth_limit,
            'entropy_cutoff': self.entropy_cutoff,
            'splitting_criterion': self.splitting_criterion,
            'min_samples_split_count': self.min_samples_split_count,
            'features': self.features
        }


    def _split(self, X_column, split_thresh):
        mask = X_column <= split_thresh
        l_idxs = np.where(mask)[0]
        r_idxs = np.where(~mask)[0]
        return l_idxs, r_idxs
    
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        current_entropy = self.criterion_function(y)

        if depth > self.depth:
            self.depth = depth
        if (self.max_depth_limit is not None and depth >= self.max_depth_limit) \
                or (self.entropy_cutoff is not None and current_entropy < self.entropy_cutoff) \
                or (num_samples < self.min_samples_split_count) \
                or (np.unique(y).size == 1):
            most_common_label = self.freq_label(y)
            return node(value=most_common_label)

        # Randomly shuffle features to determine the best split
        selected_features = np.random.choice(num_features, num_features, replace=False)
        best_feature, best_threshold = self._best_criteria(X, y, selected_features)

        # If no valid split is found, return a leaf node
        if best_feature is None:
            most_common_label = self.freq_label(y)
            return node(value=most_common_label)

        self.leaf_node_count += 1
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left_subtree = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        # Return the newly created internal node
        return node(best_feature, best_threshold, left_subtree, right_subtree)

    def _best_criteria(self, X, y, feat_idxs):
        def evaluate_split(feat_idx):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            best_local_gain = -1
            best_local_threshold = None
            
            for threshold in thresholds:
                gain = self._gain(y, X_column, threshold)
                if gain > best_local_gain:
                    best_local_gain = gain
                    best_local_threshold = threshold
            
            return feat_idx, best_local_threshold, best_local_gain

        split_results = [evaluate_split(feat_idx) for feat_idx in feat_idxs]
        best_split = max(split_results, key=lambda x: x[2])
        
        return best_split[0], best_split[1]  # Return best feature index and threshold


    def _weighted_criterion(self, y_left, y_right, criterion_function):
        total_samples = len(y_left) + len(y_right)
        if total_samples == 0:
            return 0    
        p_left = len(y_left) / total_samples
        p_right = len(y_right) / total_samples
        return p_left * criterion_function(y_left) + p_right * criterion_function(y_right)
    def _gain(self, y, X_column, split_thresh):
        l_idxs, r_idxs = self._split(X_column, split_thresh)
        
        if not l_idxs.size or not r_idxs.size:
            return 0
        
        y_left, y_right = y[l_idxs], y[r_idxs]
        
        parent_criterion = self.criterion_function(y)
        child_criterion = self._weighted_criterion(y_left, y_right, self.criterion_function)
        
        return parent_criterion - child_criterion


    # Gini impurity: For binary classification, it's 2p(1-p).
    # Here, we use the general formula for multi-class problems: 1 - sum(p_i^2)
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
   
    # Scaled entropy for binary classification: scaled_ent = - (p/2)*log2(p) - ((1-p)/2)*log2(1-p)
    # This function implements a generalized version for multi-class problems
    def _scaled_entropy(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Compute class probabilities
        total_samples = len(y)
        class_probabilities = class_counts / total_samples
        
        # In order to avoid division by zero
        safe_prob = np.maximum(class_probabilities, 1e-15)
        
        # Calculate scaled entropy
        scaled_ent = -np.sum((safe_prob / 2) * np.log2(safe_prob))
        
        return scaled_ent

    #for binary classification sqrt(p*(1-p))
    def _squared_impurity(self, y):
        # Calculate class frequencies
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Compute class probabilities
        total_samples = len(y)
        class_probabilities = class_counts / total_samples
        
        # In order to avoid division by zero
        safe_prob = np.maximum(class_probabilities, 1e-15)
        
        # Calculate squared impurity
        impurity = np.sum(np.sqrt(safe_prob * (1 - safe_prob)))
        
        return impurity


    def freq_label(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, X):
        predictions = np.empty(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            predictions[i] = self._traverse_tree(x, self.root_node)
        return predictions

# tail recursive
    def _traverse_tree(self, x, node):
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


def zero_one_loss(y_true, y_train_pred):
    return np.mean(y_train_pred != y_true)
def grid_search(X_train, y_train, param_grid, scoring_func):
    # Initialize KFold for cross-validation
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=41)

    # Refactored function to evaluate parameters
    def evaluate_params(params):
        scores, depths, leaf_counts = [], [], []

        # Iterate over each fold for cross-validation
        for train_idx, val_idx in cv_splitter.split(X_train, y_train):
            # Split training and validation data for the current fold
            X_train_cv, y_train_cv = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val_cv, y_val_cv = X_train.iloc[val_idx], y_train.iloc[val_idx]

            # Create and configure the DecisionTree model with current params
            model = DecisionTree(
                max_depth_limit=params.get('max_depth_limit'),
                entropy_cutoff=params.get('entropy_cutoff'),
                splitting_criterion=params.get('splitting_criterion'),
                min_samples_split_count=2,  # Keeping this as a constant value
                features=X_train.columns  # Pass the feature names
            )

            # Fit the model on the training fold data
            model.fit(X_train_cv.values, y_train_cv.values)

            # Predict on the validation fold
            y_pred_val = model.predict(X_val_cv.values)

            # Calculate the accuracy score using the scoring function provided
            score = scoring_func(y_val_cv, y_pred_val)

            # Accumulate results
            scores.append(score)
            depths.append(model.depth)
            leaf_counts.append(model.leaf_node_count)

        # Aggregate the results across all folds
        avg_score = np.mean(scores)
        avg_depth = np.mean(depths)
        avg_leaf_count = np.mean(leaf_counts)

        # Output current parameter results
        print(f"Params: {params} \tAccuracy: {avg_score:.5f}, Avg Depth: {avg_depth:.2f}, Avg Leafs: {avg_leaf_count:.2f}")

        return {
            'params': params,
            'score': avg_score,
            'avg_depth': avg_depth,
            'avg_leafs': avg_leaf_count
        }

    # Parallelized execution over the parameter grid
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_params)(params) for params in param_grid
    )

    # Sort results by score and extract top 5
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
    print("\nTop 5 models:")
    for rank, result in enumerate(sorted_results, 1):
       print(f"Rank {rank}:")
       print(f"  Parameters: {result['params']}")
       print(f"  Score: {result['score']:.5f}")
       print(f"  Avg Depth: {result['avg_depth']:.2f}")
       print(f"  Avg Leafs: {result['avg_leafs']:.2f}")
       print()


    return results,sorted_results[0]['params'], sorted_results[0]['score']



if __name__ == '__main__':
    
    # Load the secondary dataset
    data = pd.read_csv('secondary_data.csv', sep=';')

    # Perform one-hot encoding on categorical features and convert to integer type
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.astype(int)

    # Separate features and target
    target_column = 'class_p'
    feature_columns = [col for col in data_encoded.columns if col not in ['class_p', 'class_e']]

    X = data_encoded[feature_columns]
    y = data_encoded[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=41)

    param_grid = []
    for split_func in ['scaled_entropy', 'gini', 'squared']:
        for depth in [40, 50]:
            param_grid.append({'entropy_cutoff': 0.0001, 'splitting_criterion': split_func})
            param_grid.append({'max_depth_limit': depth, 'splitting_criterion': split_func})

    results, best_params, best_score = grid_search(X_train, y_train, param_grid, zero_one_loss)


    print(f"\nBest score: : {best_score:.5f}")
    print("Best Hyperparameters:", best_params)


    best_tree = DecisionTree(
        features=X_train.columns,
        max_depth_limit=best_params.get('max_depth_limit'),
        entropy_cutoff=best_params.get('entropy_cutoff'),
        splitting_criterion=best_params['splitting_criterion'],
    )


    best_tree.fit(X_train.values, y_train.values)

    # Make predictions on the training data
    y_train_pred = best_tree.predict(X_train.values)
    y_test_pred = best_tree.predict(X_test.values)
    # Evaluate the model on both training and testing data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_loss = zero_one_loss(y_train.values, y_train_pred)
    test_loss = zero_one_loss(y_test.values, y_test_pred)
    
    # Calculate the difference between train and test metrics
    accuracy_diff = train_accuracy - test_accuracy
    loss_diff = test_loss - train_loss
    
    print("\nModel Performance:")
    print(f"Train accuracy: {train_accuracy:.5f}")
    print(f"Test accuracy:  {test_accuracy:.5f}")
    print(f"Train zero-one loss: {train_loss:.5f}")
    print(f"Test zero-one loss:  {test_loss:.5f}")
    
    print("\nOverfitting/Underfitting Analysis:")
    if accuracy_diff > 0.05:  # This threshold can be adjusted based on your specific needs
        print(f"Potential overfitting detected. Accuracy difference: {accuracy_diff:.5f}")
    elif accuracy_diff < 0:
        print(f"Unusual behavior: Test accuracy higher than train accuracy by {-accuracy_diff:.5f}")
    else:
        print(f"Model seems well-balanced. Accuracy difference: {accuracy_diff:.5f}")
    
    if loss_diff > 0.05:  # This threshold can be adjusted based on your specific needs
        print(f"Potential overfitting detected. Zero-one loss difference: {loss_diff:.5f}")
    elif loss_diff < 0:
        print(f"Unusual behavior: Train loss higher than test loss by {-loss_diff:.5f}")
    else:
        print(f"Model seems well-balanced. Zero-one loss difference: {loss_diff:.5f}")
    
    if train_accuracy < 0.5 and test_accuracy < 0.5:
        print("Potential underfitting: Both train and test accuracies are low.")

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test.values, y_test_pred)

    # Visualize the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, cmap='YlOrRd')

    # Add text annotations
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, f'{conf_matrix[i, j]}',
                    ha='center', va='center', color='black')

    # Set up axes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")

    # Set title and labels
    ax.set_title('Confusion Matrix Visualization')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')

    plt.tight_layout()
    plt.show()

# Thank you Beyonce


