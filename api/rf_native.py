import struct


class NativeRandomForest:
    """Compact evaluator for exported sklearn RandomForestRegressor trees."""

    HEADER = b'RFB1'
    NODE = struct.Struct('<iiiff')

    def __init__(self, path):
        with open(path, 'rb') as f:
            if f.read(4) != self.HEADER:
                raise ValueError('Invalid Random Forest model file')
            self.num_features, tree_count = struct.unpack('<II', f.read(8))
            self.trees = []
            for _ in range(tree_count):
                (node_count,) = struct.unpack('<I', f.read(4))
                nodes = [
                    self.NODE.unpack(f.read(self.NODE.size))
                    for _ in range(node_count)
                ]
                self.trees.append(nodes)

    @staticmethod
    def _predict_tree(tree, features):
        node = 0
        while tree[node][0] != -1:
            left, right, feature, threshold, value = tree[node]
            node = left if features[feature] <= threshold else right
        return tree[node][4]

    def predict(self, features):
        return sum(self._predict_tree(tree, features) for tree in self.trees) / len(self.trees)

    def shap_values(self, features):
        baseline = self.predict([0.0] * self.num_features)
        values = []
        for index, feature in enumerate(features):
            changed = list(features)
            changed[index] = 0.0
            values.append(self.predict(changed) - baseline)
        return values
