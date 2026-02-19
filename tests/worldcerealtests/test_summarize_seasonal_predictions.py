import unittest

import numpy as np
import torch

from worldcereal.train.finetuning_utils import summarize_seasonal_predictions
from worldcereal.train.seasonal_head import SeasonalHeadOutput


class TestSummarizeSeasonalPredictions(unittest.TestCase):
    def test_croptype_samples_also_contribute_landcover_metrics(self):
        output = SeasonalHeadOutput(
            global_logits=torch.tensor(
                [
                    [4.0, 1.0],
                    [1.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            season_logits=torch.tensor(
                [
                    [[3.0, 1.0]],
                    [[1.0, 3.0]],
                ],
                dtype=torch.float32,
            ),
            global_embedding=torch.zeros(2, 4),
            season_embeddings=torch.zeros(2, 1, 4),
            season_masks=torch.ones(2, 1, 1, dtype=torch.bool),
        )

        attrs = {
            "landcover_label": ["temporary_crops", "temporary_crops"],
            "croptype_label": [None, "wheat"],
            "label_task": ["landcover", "croptype"],
            "season_masks": np.ones((2, 1, 1), dtype=bool),
            "in_seasons": np.ones((2, 1), dtype=bool),
            "valid_position": [0, 0],
        }

        summary = summarize_seasonal_predictions(
            output,
            attrs,
            landcover_classes=["temporary_crops", "water"],
            croptype_classes=["wheat", "maize"],
            cropland_class_names=["temporary_crops"],
        )

        self.assertEqual(
            len(summary["landcover"]),
            2,
            "Landcover metrics should include croptype-supervised samples.",
        )
        self.assertEqual(
            len(summary["croptype"]),
            1,
            "Only croptype-supervised samples should contribute to crop-type metrics.",
        )


if __name__ == "__main__":
    unittest.main()
