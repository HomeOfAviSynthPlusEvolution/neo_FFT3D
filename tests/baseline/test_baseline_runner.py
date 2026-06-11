import hashlib
import unittest

from tests.baseline import baseline_runner


class CanonicalHashTests(unittest.TestCase):
    def test_hashes_only_valid_row_bytes(self):
        planes = [
            baseline_runner.PlaneBytes(
                name="Y",
                width=3,
                height=2,
                bytes_per_sample=1,
                stride=5,
                data=b"abcXXdefYY",
            )
        ]

        result = baseline_runner.hash_planes(planes)

        self.assertEqual(result, hashlib.sha256(b"abcdef").hexdigest())

    def test_hashes_planes_in_supplied_order(self):
        planes = [
            baseline_runner.PlaneBytes("Y", 2, 1, 1, 2, b"ab"),
            baseline_runner.PlaneBytes("U", 2, 1, 1, 2, b"cd"),
            baseline_runner.PlaneBytes("V", 2, 1, 1, 2, b"ef"),
        ]

        result = baseline_runner.hash_planes(planes)

        self.assertEqual(result, hashlib.sha256(b"abcdef").hexdigest())


class CaseSelectionTests(unittest.TestCase):
    def test_filters_cases_by_tier(self):
        cases = [
            {"id": "smoke_default", "tier": "smoke"},
            {"id": "compat_yuv", "tier": "compat"},
            {"id": "full_hd", "tier": "full"},
        ]

        selected = baseline_runner.select_cases(cases, "smoke")

        self.assertEqual([case["id"] for case in selected], ["smoke_default"])

    def test_all_tier_keeps_all_cases(self):
        cases = [
            {"id": "smoke_default", "tier": "smoke"},
            {"id": "compat_yuv", "tier": "compat"},
        ]

        selected = baseline_runner.select_cases(cases, "all")

        self.assertEqual([case["id"] for case in selected], ["smoke_default", "compat_yuv"])


class ScriptRenderingTests(unittest.TestCase):
    def test_renders_vs_filter_call_with_arrays_and_strings(self):
        source = {
            "type": "blank",
            "format": "GRAY8",
            "width": 64,
            "height": 48,
            "length": 9,
            "color": [64],
        }
        params = {"bt": 3, "sigma": 2.0, "planes": [0], "opt": 1}

        script = baseline_runner.render_vs_script("/tmp/plugin.so", source, params)

        self.assertIn('core.std.LoadPlugin(path="/tmp/plugin.so")', script)
        self.assertIn("core.std.BlankClip(width=64, height=48, length=9, format=vs.GRAY8, color=[64])", script)
        self.assertIn('core.neo_fft3d.FFT3D(src, bt=3, opt=1, planes=[0], sigma=2.0)', script)

    def test_renders_avs_filter_call_with_arrays_as_strings(self):
        source = {
            "type": "blank",
            "format": "Y8",
            "width": 64,
            "height": 48,
            "length": 9,
            "color": 64,
        }
        params = {"bt": 3, "sigma": 2.0, "y": 3, "u": 2, "v": 2}

        script = baseline_runner.render_avs_script("/tmp/plugin.so", source, params)

        self.assertIn('LoadPlugin("/tmp/plugin.so")', script)
        self.assertIn('src = BlankClip(width=64, height=48, length=9, pixel_type="Y8", color=64)', script)
        self.assertIn('return neo_fft3d(src, bt=3, sigma=2.0, u=2, v=2, y=3)', script)
