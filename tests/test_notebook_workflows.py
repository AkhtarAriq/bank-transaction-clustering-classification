"""Regression tests for notebook data validation, transforms, and evaluation.

Run with ``python -m unittest discover -s tests -v`` from the project root.
Only cells tagged ``definitions`` are loaded; training and exports never run here.
"""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLUSTERING_NOTEBOOK = ROOT / "[Clustering]_Submission_Akhir_BMLP_Muhammad_Akhtar_Ariq.ipynb"
CLASSIFICATION_NOTEBOOK = ROOT / "[Klasifikasi]_Submission_Akhir_BMLP_Muhammd_Akhtar_Ariq.ipynb"


def load_definitions(path):
    """Load the actual notebook helpers without executing its analysis cells."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "definitions" in cell["metadata"].get("tags", []):
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    return namespace


def transaction_fixture():
    """Return varied, valid records with known original numerical values."""
    count = 12
    return pd.DataFrame({
        "TransactionID": [f"TX{i:03}" for i in range(count)],
        "AccountID": [f"AC{i:03}" for i in range(count)],
        "TransactionAmount": np.arange(10.0, 130.0, 10.0),
        "PreviousTransactionDate": ["2024-01-01"] * count,
        "TransactionType": ["Credit", "Debit"] * 6,
        "Location": ["Austin", "Boston", "Chicago"] * 4,
        "DeviceID": [f"D{i:03}" for i in range(count)],
        "IP Address": ["192.0.2.1"] * count,
        "MerchantID": ["M001"] * count,
        "Channel": ["ATM", "Branch", "Online"] * 4,
        "CustomerAge": np.arange(20.0, 80.0, 5.0),
        "CustomerOccupation": ["Doctor", "Engineer", "Student"] * 4,
        "TransactionDuration": np.arange(10.0, 130.0, 10.0),
        "LoginAttempts": [1.0] * count,
        "AccountBalance": np.arange(100.0, 1300.0, 100.0),
        "TransactionDate": ["2024-01-02"] * count,
    })


class NotebookWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clustering = load_definitions(CLUSTERING_NOTEBOOK)
        cls.classification = load_definitions(CLASSIFICATION_NOTEBOOK)

    def helper(self, namespace, name):
        function = namespace.get(name)
        self.assertTrue(callable(function), f"Notebook must expose the documented {name} helper")
        return function

    def test_missing_transaction_feature_is_rejected(self):
        validate = self.helper(self.clustering, "validate_transactions")
        with self.assertRaisesRegex(ValueError, "AccountBalance"):
            validate(transaction_fixture().drop(columns="AccountBalance"))

    def test_infinite_transaction_value_is_rejected(self):
        validate = self.helper(self.clustering, "validate_transactions")
        frame = transaction_fixture()
        frame.loc[0, "TransactionAmount"] = np.inf
        with self.assertRaises(ValueError):
            validate(frame)

    def test_missing_values_are_allowed_before_cleaning(self):
        validate = self.helper(self.clustering, "validate_transactions")
        frame = transaction_fixture()
        frame.loc[0, "CustomerAge"] = np.nan
        validate(frame)

    def test_iqr_filter_removes_extreme_value_without_mutating_input(self):
        filter_outliers = self.helper(self.clustering, "filter_iqr_outliers")
        original = pd.DataFrame({"amount": [10, 11, 12, 13, 1000]}, index=[5, 6, 7, 8, 9])
        snapshot = original.copy(deep=True)
        result = filter_outliers(original, ["amount"])
        self.assertEqual(result.index.tolist(), [5, 6, 7, 8])
        pd.testing.assert_frame_equal(original, snapshot)

    def test_zero_iqr_retains_only_the_constant_value(self):
        filter_outliers = self.helper(self.clustering, "filter_iqr_outliers")
        result = filter_outliers(pd.DataFrame({"attempts": [1, 1, 1, 1, 5]}), ["attempts"])
        self.assertEqual(result["attempts"].tolist(), [1, 1, 1, 1])

    def test_feature_transforms_can_recover_original_values(self):
        encode = self.helper(self.clustering, "encode_features")
        original = transaction_fixture()
        features, scaler, encoders, age_edges = encode(original)
        numerical = list(self.clustering["NUMERICAL_FEATURES"])
        np.testing.assert_allclose(scaler.inverse_transform(features[numerical]), original[numerical])
        for column in ["TransactionType", "Location", "Channel", "CustomerOccupation"]:
            recovered = encoders[column].inverse_transform(features[column])
            self.assertEqual(recovered.tolist(), original[column].tolist())
        self.assertEqual(features.index.tolist(), original.index.tolist())
        self.assertEqual(features["AgeGroup"].nunique(), 3)
        self.assertEqual(len(age_edges), 4)
        self.assertNotIn("Target", features.columns)

    def test_missing_classification_target_is_rejected(self):
        validate = self.helper(self.classification, "validate_labeled_data")
        with self.assertRaisesRegex(ValueError, "Target"):
            validate(transaction_fixture())

    def test_invalid_cluster_label_is_rejected(self):
        validate = self.helper(self.classification, "validate_labeled_data")
        frame = transaction_fixture().assign(AgeGroup="Muda", Target=[0, 1, 2] * 4)
        frame.loc[0, "Target"] = 9
        with self.assertRaises(ValueError):
            validate(frame)

    def test_metrics_match_hand_calculated_multiclass_predictions(self):
        evaluate = self.helper(self.classification, "evaluate_predictions")
        result = evaluate([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["precision_weighted"], 5 / 6)
        self.assertAlmostEqual(result["recall_weighted"], 0.75)
        self.assertAlmostEqual(result["f1_weighted"], 11 / 15)


if __name__ == "__main__":
    unittest.main()
