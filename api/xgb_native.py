import json
import math


class NativeXGBModel:
    """Minimal evaluator for the numeric gbtree models produced by XGBoost."""

    def __init__(self, path):
        with open(path, 'r') as f:
            document = json.load(f)

        learner = document['learner']
        self.objective = learner['objective']['name']
        params = learner['learner_model_param']
        self.num_features = int(params['num_feature'])
        base_score = float(params['base_score'].strip('[]').split(',')[0])
        self.base_score = (
            math.log(base_score / (1.0 - base_score))
            if self.objective == 'binary:logistic'
            else base_score
        )
        self.trees = learner['gradient_booster']['model']['trees']

    @staticmethod
    def _tree_predict(tree, features):
        node = 0
        while tree['left_children'][node] != -1:
            feature_index = tree['split_indices'][node]
            threshold = tree['split_conditions'][node]
            value = features[feature_index]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                node = (
                    tree['left_children'][node]
                    if tree['default_left'][node]
                    else tree['right_children'][node]
                )
            elif value < threshold:
                node = tree['left_children'][node]
            else:
                node = tree['right_children'][node]
        return tree['base_weights'][node]

    def predict(self, features):
        margin = self.base_score + sum(
            self._tree_predict(tree, features) for tree in self.trees
        )
        if self.objective == 'binary:logistic':
            margin = max(-709.0, min(709.0, margin))
            return 1.0 / (1.0 + math.exp(-margin))
        return margin

    @staticmethod
    def _extend_path(path, depth, zero_fraction, one_fraction, feature_index):
        path = [item.copy() for item in path]
        item = {
            'feature_index': feature_index,
            'zero_fraction': zero_fraction,
            'one_fraction': one_fraction,
            'pweight': 1.0 if depth == 0 else 0.0,
        }
        if len(path) <= depth:
            path.append(item)
        else:
            path[depth] = item
        for i in range(depth - 1, -1, -1):
            path[i + 1]['pweight'] += (
                one_fraction * path[i]['pweight'] * (i + 1) / (depth + 1)
            )
            path[i]['pweight'] = (
                zero_fraction * path[i]['pweight'] * (depth - i) / (depth + 1)
            )
        return path

    @staticmethod
    def _unwind_path(path, depth, path_index):
        path = [item.copy() for item in path]
        one_fraction = path[path_index]['one_fraction']
        zero_fraction = path[path_index]['zero_fraction']
        next_one_portion = path[depth]['pweight']

        for i in range(depth - 1, -1, -1):
            if one_fraction != 0:
                old_weight = path[i]['pweight']
                path[i]['pweight'] = (
                    next_one_portion * (depth + 1) / ((i + 1) * one_fraction)
                )
                next_one_portion = (
                    old_weight
                    - path[i]['pweight'] * zero_fraction * (depth - i)
                    / (depth + 1)
                )
            else:
                path[i]['pweight'] = (
                    path[i]['pweight'] * (depth + 1)
                    / (zero_fraction * (depth - i))
                )

        path.pop(path_index)
        return path

    @staticmethod
    def _unwound_path_sum(path, depth, path_index):
        one_fraction = path[path_index]['one_fraction']
        zero_fraction = path[path_index]['zero_fraction']
        next_one_portion = path[depth]['pweight']
        total = 0.0

        if one_fraction != 0:
            for i in range(depth - 1, -1, -1):
                tmp = next_one_portion / ((i + 1) * one_fraction)
                total += tmp
                next_one_portion = (
                    path[i]['pweight']
                    - tmp * zero_fraction * (depth - i)
                )
        else:
            for i in range(depth - 1, -1, -1):
                total += path[i]['pweight'] / (zero_fraction * (depth - i))
        return total * (depth + 1)

    def _tree_shap(self, tree, features):
        values = [0.0] * self.num_features

        def recurse(node, unique_depth, parent_path, parent_zero, parent_one, parent_feature):
            path = [item.copy() for item in parent_path]
            path = self._extend_path(
                path, unique_depth, parent_zero, parent_one, parent_feature
            )

            left = tree['left_children'][node]
            right = tree['right_children'][node]
            if left == -1:
                leaf_value = tree['base_weights'][node]
                for i in range(1, unique_depth + 1):
                    feature_index = path[i]['feature_index']
                    if feature_index < 0:
                        continue
                    weight = self._unwound_path_sum(path, unique_depth, i)
                    values[feature_index] += (
                        weight
                        * (path[i]['one_fraction'] - path[i]['zero_fraction'])
                        * leaf_value
                    )
                return

            split_index = tree['split_indices'][node]
            hot = left
            cold = right
            value = features[split_index]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                if not tree['default_left'][node]:
                    hot, cold = cold, hot
            elif value > tree['split_conditions'][node]:
                hot, cold = right, left

            node_cover = tree['sum_hessian'][node] or 1.0
            hot_zero = tree['sum_hessian'][hot] / node_cover
            cold_zero = tree['sum_hessian'][cold] / node_cover
            previous = next(
                (i for i, item in enumerate(path)
                 if item['feature_index'] == split_index),
                None,
            )
            incoming_zero = 1.0
            incoming_one = 1.0
            if previous is not None:
                incoming_zero = path[previous]['zero_fraction']
                incoming_one = path[previous]['one_fraction']
                path = self._unwind_path(path, unique_depth, previous)
                unique_depth -= 1

            recurse(
                hot, unique_depth + 1, path,
                hot_zero * incoming_zero, incoming_one, split_index
            )
            recurse(
                cold, unique_depth + 1, path,
                cold_zero * incoming_zero, 0.0, split_index
            )

        recurse(
            0,
            0, [], 1.0, 1.0, -1,
        )
        return values

    def shap_values(self, features):
        values = [0.0] * self.num_features
        for tree in self.trees:
            tree_values = self._tree_shap(tree, features)
            for index, value in enumerate(tree_values):
                values[index] += value
        return values
