from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from evaluation.publication_protocol import OBJECTIVES, SEEDS, SUBJECTS, build_publication_runs, cross_subject_fold
from models.losses import hybrid_eeg_anchor_contrastive_loss
from scripts.publication_analysis import build_analysis, exact_sign_flip_pvalue, holm_adjust
from training.datasets import validate_anchor_image_scope


class PublicationProtocolTests(unittest.TestCase):
    def test_cross_subject_folds_are_disjoint(self) -> None:
        for subject in SUBJECTS:
            train, val, test = cross_subject_fold(subject)
            self.assertEqual(len(train), 8)
            self.assertNotIn(val, train)
            self.assertNotIn(test, train)
            self.assertNotEqual(val, test)
            self.assertEqual(set(train) | {val, test}, set(SUBJECTS))

    def test_full_protocol_has_expected_unique_runs(self) -> None:
        runs = build_publication_runs()
        expected = len(SUBJECTS) * len(SEEDS) * (len(OBJECTIVES) + 1)
        self.assertEqual(len(runs), expected)
        self.assertEqual(len({run.experiment_id for run in runs}), expected)
        self.assertIn("hybrid", OBJECTIVES)

    def test_pilot_protocol_accepts_custom_seed_set(self) -> None:
        runs = build_publication_runs(
            subjects=("sub-10",),
            seeds=(42, 512, 1024),
            objectives=("direct", "anchor", "hybrid"),
            include_subject_specific=False,
        )
        self.assertEqual(len(runs), 9)
        self.assertIn("CLEAN-SUB-10-HYBRID-S1024", {run.experiment_id for run in runs})

    def test_hybrid_loss_has_finite_gradients(self) -> None:
        torch.manual_seed(42)
        eeg_embeddings = torch.randn(4, 512, requires_grad=True)
        image_embeddings = torch.randn(4, 512)
        anchor_embeddings = torch.randn(3, 512)
        target_ids = torch.tensor([0, 0, 1, 2])
        anchor_indices = torch.tensor([0, 0, 1, 2])

        loss = hybrid_eeg_anchor_contrastive_loss(
            eeg_embeddings,
            image_embeddings,
            anchor_embeddings,
            target_ids,
            anchor_indices,
            lambda_anchor=0.2,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(eeg_embeddings.grad)
        self.assertTrue(torch.isfinite(eeg_embeddings.grad).all().item())

    def test_anchor_scope_requires_exact_train_membership(self) -> None:
        cache = {
            "exemplar_image_indices": np.asarray(
                [np.asarray([0, 1]), np.asarray([2, 3])],
                dtype=object,
            )
        }
        validate_anchor_image_scope(
            cache,
            allowed_image_indices={0, 1, 2, 3},
            forbidden_image_indices={4},
        )
        with self.assertRaises(ValueError):
            validate_anchor_image_scope(
                cache,
                allowed_image_indices={0, 1, 2},
                forbidden_image_indices={3},
            )

    def test_sign_flip_and_holm_are_bounded(self) -> None:
        pvalue = exact_sign_flip_pvalue(np.asarray([1.0, 1.0, 1.0, 1.0]))
        self.assertEqual(pvalue, 0.125)
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
        self.assertEqual(adjusted["a"], 0.03)
        self.assertEqual(adjusted["b"], 0.08)
        self.assertEqual(adjusted["c"], 0.20)

    def test_partial_analysis_requires_complete_seed_sets_for_pairing(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "setting": "cross_subject",
                    "test_subject": "sub-01",
                    "objective": "direct",
                    "seed": seed,
                    "mrr": 0.08,
                    "top1": 0.02,
                    "top5": 0.10,
                    "top10": 0.17,
                }
                for seed in (42, 123, 256)
            ]
            + [
                {
                    "setting": "cross_subject",
                    "test_subject": "sub-01",
                    "objective": "hybrid",
                    "seed": 42,
                    "mrr": 0.081,
                    "top1": 0.021,
                    "top5": 0.101,
                    "top10": 0.171,
                }
            ]
        )
        subject_means, tests, _ = build_analysis(metrics)

        hybrid_row = subject_means.loc[subject_means["objective"] == "hybrid"].iloc[0]
        self.assertEqual(hybrid_row["seed_count"], 1)
        self.assertFalse(bool(hybrid_row["complete_seed_set"]))
        self.assertTrue((tests["n_subjects"] == 0).all())


if __name__ == "__main__":
    unittest.main()
