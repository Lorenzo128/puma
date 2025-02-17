#!/usr/bin/env python
"""
Unit test script for the functions in hlplots/tagger.py
"""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from puma.hlplots import Results
from puma.hlplots.tagger import Tagger
from puma.utils import (
    get_dummy_2_taggers,
    get_dummy_multiclass_scores,
    logger,
    set_log_level,
)

set_log_level(logger, "DEBUG")


class ResultsTestCase(unittest.TestCase):
    """Test class for the Results class."""

    def test_add_duplicated(self):
        """Test empty string as model name."""
        dummy_tagger_1 = Tagger("dummy")
        dummy_tagger_2 = Tagger("dummy")
        results = Results()
        results.add(dummy_tagger_1)
        with self.assertRaises(KeyError):
            results.add(dummy_tagger_2)

    def test_add_2_taggers(self):
        """Test empty string as model name."""
        dummy_tagger_1 = Tagger("dummy")
        dummy_tagger_2 = Tagger("dummy_2")
        results = Results()
        results.add(dummy_tagger_1)
        results.add(dummy_tagger_2)
        self.assertEqual(
            list(results.taggers.keys()),
            ["dummy (dummy)", "dummy_2 (dummy_2)"],  # pylint: disable=W0212
        )

    def test_get_taggers(self):
        """Test empty string as model name."""
        dummy_tagger_1 = Tagger("dummy")
        dummy_tagger_2 = Tagger("dummy_2")
        results = Results()
        results.add(dummy_tagger_1)
        results.add(dummy_tagger_2)
        retrieved_dummy_tagger_2 = results["dummy_2 (dummy_2)"]
        self.assertEqual(retrieved_dummy_tagger_2.name, dummy_tagger_2.name)

    def test_add_taggers_from_file(self):
        """Test for Results.add_taggers_from_file function"""
        tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        rng = np.random.default_rng(seed=16)
        with h5py.File(f"{tmp_dir.name}/test.h5", "w") as file:
            data = get_dummy_2_taggers()
            data["pt"] = rng.random(len(data))
            file.create_dataset("jets", data=data.to_records())
        results = Results()
        taggers = [Tagger("rnnip")]
        results.add_taggers_from_file(
            taggers, f"{tmp_dir.name}/test.h5", perf_var=data["pt"]
        )
        self.assertEqual(list(results.taggers.values()), taggers)


class ResultsPlotsTestCase(unittest.TestCase):
    """Test class for the Results class running plots."""

    def setUp(self) -> None:
        """Set up for unit tests."""
        scores, labels = get_dummy_multiclass_scores()
        tagger_args = {
            "is_flav": {
                "ujets": labels == 0,
                "cjets": labels == 4,
                "bjets": labels == 5,
            }
        }
        dummy_tagger_1 = Tagger("dummy", **tagger_args)
        dummy_tagger_1.scores = scores
        dummy_tagger_1.label = "dummy tagger"
        self.dummy_tagger_1 = dummy_tagger_1

    def assertIsFile(self, path: str):  # pylint: disable=invalid-name
        """Check for file to exist.
        Taken from https://stackoverflow.com/a/59198749/10896585
        Parameters
        ----------
        path : str
            Path to file

        Raises
        ------
        AssertionError
            if file does not exist
        """
        if not Path(path).resolve().is_file():
            raise AssertionError(f"File does not exist: {path}")

    def test_plot_roc_bjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.f_c = 0.05
        results = Results(signal="bjets")
        results.add(self.dummy_tagger_1)
        results.sig_eff = np.linspace(0.6, 0.95, 20)
        with tempfile.TemporaryDirectory() as tmp_file:
            plot_name = f"{tmp_file}/dummy_plot.png"
            results.plot_rocs(plot_name=plot_name)
            self.assertIsFile(plot_name)

    def test_plot_roc_cjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.f_b = 0.05
        results = Results(signal="cjets")
        results.add(self.dummy_tagger_1)
        results.sig_eff = np.linspace(0.2, 0.95, 20)
        with tempfile.TemporaryDirectory() as tmp_file:
            plot_name = f"{tmp_file}/dummy_plot.png"
            results.plot_rocs(plot_name=plot_name)
            self.assertIsFile(plot_name)

    def test_plot_var_perf_bjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.f_c = 0.05
        self.dummy_tagger_1.disc_cut = 2
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_var = rng.exponential(
            100, size=len(self.dummy_tagger_1.scores)
        )
        results = Results(signal="bjets")
        results.add(self.dummy_tagger_1)
        with tempfile.TemporaryDirectory() as tmp_file:
            plot_name = f"{tmp_file}/dummy_plot"
            results.plot_var_perf(
                plot_name=plot_name,
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
            )
            self.assertIsFile(plot_name + "_bjets_eff.png")
            self.assertIsFile(plot_name + "_cjets_rej.png")
            self.assertIsFile(plot_name + "_ujets_rej.png")

    def test_plot_var_perf_cjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.f_b = 0.05
        self.dummy_tagger_1.working_point = 0.5
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_var = rng.exponential(
            100, size=len(self.dummy_tagger_1.scores)
        )
        results = Results(signal="cjets")
        results.add(self.dummy_tagger_1)
        with tempfile.TemporaryDirectory() as tmp_file:
            plot_name = f"{tmp_file}/dummy_plot"
            results.plot_var_perf(
                plot_name=plot_name,
                h_line=self.dummy_tagger_1.working_point,
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
            )
            self.assertIsFile(plot_name + "_cjets_eff.png")
            self.assertIsFile(plot_name + "_bjets_rej.png")
            self.assertIsFile(plot_name + "_ujets_rej.png")

    def test_plot_discs_bjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.f_c = 0.05
        results = Results(signal="bjets")
        results.add(self.dummy_tagger_1)
        with tempfile.TemporaryDirectory() as tmp_file:
            plot_name = f"{tmp_file}/dummy_plot.png"
            results.plot_discs(plot_name=plot_name)
            self.assertIsFile(plot_name)

    def test_plot_discs_cjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.f_b = 0.05
        results = Results(signal="cjets")
        results.add(self.dummy_tagger_1)
        with tempfile.TemporaryDirectory() as tmp_file:
            plot_name = f"{tmp_file}/dummy_plot.png"
            results.plot_discs(plot_name=plot_name)
            self.assertIsFile(plot_name)
