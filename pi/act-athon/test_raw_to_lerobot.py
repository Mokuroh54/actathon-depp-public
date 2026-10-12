import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from lerobot.datasets import LeRobotDataset

from raw_to_lerobot import find_episodes, load_episode, load_outcomes, main, stale_reason, validate_video_counts


class RawEpisodesTest(unittest.TestCase):
    def make_episode(self, path):
        path.mkdir(parents=True)
        (path / "meta.json").write_text(json.dumps({"fps": 15}))
        (path / "steps.jsonl").write_text('{"i": 0}\n{"i": 2}\n')
        (path / "front").mkdir()
        return path.resolve()

    def test_selects_saved_episodes_across_roots_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = self.make_episode(root / "archive" / "session1" / "saved")
            self.make_episode(root / "live" / "recording.partial")
            current = self.make_episode(root / "live" / "saved2")
            self.assertEqual(find_episodes([root, root / "archive"], ["front"]),
                             sorted([saved, current]))
            self.assertEqual(find_episodes([root / "live" / "recording.partial"], ["front"]), [])

    def test_include_applies_to_local_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = self.make_episode(root / "session1" / "saved")
            self.make_episode(root / "session2" / "saved")
            self.assertEqual(find_episodes([root], ["front"], "session1/*"), [selected])
            self.assertEqual(find_episodes([root], ["front", "wrist"]), [])

    def test_missing_images_do_not_truncate_later_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = self.make_episode(Path(tmp) / "saved")
            (ep / "front" / "000002.jpg").touch()
            _, steps, n = load_episode(ep, ["front"])
            self.assertEqual([s["i"] for s in steps], [0, 2])
            self.assertEqual(n, 2)

    def test_stale_frames_and_state_reject(self):
        fresh = {"i": 0, "state": {"poll_age": 0.1}, "cams": {"front": {"age": 0.2}}}
        self.assertIsNone(stale_reason([fresh, {"i": 1}], ["front"], 0.5))   # fields absent: nothing to check
        self.assertIn("front frame", stale_reason([{"i": 3, "cams": {"front": {"age": 0.7}}}], ["front"], 0.5))
        self.assertIn("gate state", stale_reason([{"i": 3, "state": {"poll_age": 0.9}}], ["front"], 0.5))

    def test_missing_root_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "does not exist"):
                find_episodes([Path(tmp) / "missing"], ["front"])

    def test_outcomes_require_unique_complete_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            row = {"source_episode_id": "one", "outcome": "uncertain"}
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(load_outcomes(path, ["one"]), {"one": row})
            with self.assertRaisesRegex(ValueError, "missing outcome"):
                load_outcomes(path, ["two"])
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate outcome"):
                load_outcomes(path, ["one"])

    def test_low_memory_lerobot_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep = self.make_episode(root / "raw" / "saved")
            (ep / "meta.json").write_text(json.dumps({"fps": 15, "task": "test"}))
            steps = [{"i": i, "action": {"swing": i / 10}} for i in range(3)]
            (ep / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps))
            for i in [0, 2]:
                cv2.imwrite(str(ep / "front" / f"{i:06d}.jpg"),
                            np.full((64, 80, 3), 40 + i, dtype=np.uint8))
            output = root / "converted"
            outcomes = root / "outcomes.jsonl"
            outcomes.write_text(json.dumps({"source_episode_id": "saved", "outcome": "uncertain",
                                             "method": "test", "requires_review": True}) + "\n")
            main(SimpleNamespace(raw_dir=[str(root / "raw")], include=None, cams=["front"],
                                 root=str(output), out_repo="local/test", low_memory=True, push=False,
                                 center_crop=True, max_age=0.5, video_preset="veryfast", outcomes=str(outcomes)))
            ds = LeRobotDataset("local/test", root=output, video_backend="pyav")
            self.assertEqual(ds.num_episodes, 1)
            self.assertEqual(ds.num_frames, 2)
            self.assertAlmostEqual(ds[1]["action"][0].item(), 0.2)
            # state = the previous STEP's command (the runner's), not the frame's own action; step 1 was skipped
            self.assertAlmostEqual(ds[0]["observation.state"][0].item(), 0.0)
            self.assertAlmostEqual(ds[1]["observation.state"][0].item(), 0.1)
            self.assertEqual(json.loads((output / "crops.json").read_text()), {"front": [8, 0, 64]})
            self.assertEqual(tuple(ds[1]["observation.images.front"].shape), (3, 64, 64))
            self.assertTrue(list(output.rglob("*.mp4")))
            self.assertTrue(list(output.rglob("*.parquet")))
            with self.assertRaisesRegex(ValueError, "video has 2 frames; expected 3"):
                validate_video_counts(output, ["front"], 3)
            sources = json.loads((output / "source_episodes.json").read_text())
            self.assertEqual(sources["saved"]["source_frame_indices"], [0, 2])
            self.assertEqual(sources["saved"]["crops"]["front"], [8, 0, 64])
            outcome = json.loads((output / "meta" / "episode_outcomes.jsonl").read_text())
            self.assertEqual(outcome["episode_index"], 0)
            self.assertEqual(outcome["outcome"], "uncertain")
            all_sources = json.loads((output / "meta" / "source_outcomes.jsonl").read_text())
            self.assertTrue(all_sources["included_in_training"])


if __name__ == "__main__":
    unittest.main()
