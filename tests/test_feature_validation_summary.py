"""CPU-only integrity checks for reports and decoded-media comparisons."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.summarize_feature_validation import summarize

ROOT = Path(__file__).resolve().parents[1]


class SummaryIntegrityTests(unittest.TestCase):
    def test_exit_record_must_match_current_comparison(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            report = root / 'case-comparison.json'
            process = root / 'case-process.json'
            report.write_text(json.dumps({'comparison': {'passed': True}}))
            self.assertEqual(summarize(root)['reports']['case']['status'], 'exit unrecorded')
            process.write_text(json.dumps({'exit_code': 0, 'comparison_sha256': '0' * 64}))
            self.assertEqual(summarize(root)['reports']['case']['status'], 'process record stale')
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            process.write_text(json.dumps({'exit_code': 0, 'comparison_sha256': digest}))
            # Matching content is reliable even if timestamps were preserved on copy.
            os.utime(process, ns=(1, 1))
            self.assertEqual(summarize(root)['reports']['case']['status'], 'passed')
            process.write_text(json.dumps({'exit_code': 0}))
            os.utime(process, ns=(1, 1))
            self.assertEqual(summarize(root)['reports']['case']['status'], 'process record stale')
            timestamp = report.stat().st_mtime_ns + 1000000
            os.utime(process, ns=(timestamp, timestamp))
            self.assertEqual(summarize(root)['reports']['case']['status'], 'passed')

    def test_two_decode_failures_cannot_report_structural_equivalence(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            for side in ('app', 'native'):
                (root / ('case-' + side + '.mp4')).write_bytes(b'invalid')
            with patch('tools.summarize_feature_validation.media', side_effect=ValueError('bad media')):
                pair = summarize(root)['media_pairs']['case-app.mp4']
            self.assertFalse(pair['structural_counts_match'])
            self.assertFalse(pair['both_videos_decoded'])
            valid = ({'video': {'width': 32, 'height': 32, 'frames': 2, 'fps': 24}}, [])
            with patch('tools.summarize_feature_validation.media', return_value=valid):
                self.assertTrue(summarize(root)['media_pairs']['case-app.mp4']['structural_counts_match'])


if __name__ == '__main__':
    unittest.main()
