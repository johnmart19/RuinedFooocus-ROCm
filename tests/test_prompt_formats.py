"""One Button prompt formats per model family, without loading models."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from random_prompt import prompt_formats as pf  # noqa: E402


class FamilyTests(unittest.TestCase):
    def test_checkpoint_families_pick_their_format(self):
        cases = {
            "Anima": pf.ANIMA, "Illustrious": pf.ANIME, "NoobAI": pf.ANIME, "Pony": pf.ANIME,
            "SDXL 1.0": pf.TAGS, "SD 1.5": pf.TAGS, "Flux.1 D": pf.NATURAL, "Qwen": pf.NATURAL,
            "Krea 2": pf.NATURAL, "MOSS Image": pf.NATURAL, None: pf.TAGS, "unknown": pf.TAGS,
        }
        for family, expected in cases.items():
            self.assertEqual(pf.family_format(family), expected, family)

    def test_explicit_choices_win_and_old_or_bad_values_stay_valid(self):
        self.assertEqual(pf.resolve(pf.ANIME, "Flux.1 D"), pf.ANIME)
        self.assertEqual(pf.resolve(pf.AUTO, "Anima"), pf.ANIMA)
        self.assertEqual(pf.resolve("Stable Cascade"), pf.NATURAL)
        self.assertEqual(pf.resolve("removed format"), pf.TAGS)

    def test_auto_without_checkpoint_lookup_falls_back_to_tags(self):
        with mock.patch.object(pf, "checkpoint_family", return_value=None):
            self.assertEqual(pf.format_for(pf.AUTO, "missing.safetensors"), pf.TAGS)
        self.assertEqual(pf.engine_mode(pf.ANIMA), pf.ANIME)
        self.assertEqual(pf.engine_mode(pf.NATURAL), pf.NATURAL)


class FinalizeTests(unittest.TestCase):
    RAW = "(1girl, solo, Tomoko:1.3), awardwinning photo 😊, [blue_hair], 8k,  , sharp focus <lora:detail:0.6>"

    def test_tags_keep_weights_and_lora_tags(self):
        out = pf.finalize(self.RAW, pf.TAGS)
        self.assertIn("(1girl, solo, Tomoko:1.3)", out)
        self.assertIn("award-winning", out)
        self.assertNotIn("😊", out)
        self.assertNotIn(", ,", out)
        self.assertTrue(out.endswith("<lora:detail:0.6>"))

    def test_anime_adds_quality_tags_once(self):
        out = pf.finalize("masterpiece, 1girl, smile", pf.ANIME)
        self.assertTrue(out.startswith("best quality, very aesthetic, absurdres, masterpiece, 1girl"))
        self.assertEqual(out.count("masterpiece"), 1)

    def test_anima_is_lowercase_without_weights_or_underscores(self):
        out = pf.finalize(self.RAW + ", score_9", pf.ANIMA)
        self.assertTrue(out.startswith("masterpiece, best quality, 1girl, solo, tomoko"))
        text = out.split("<")[0]
        self.assertNotRegex(text, r"[()\[\]:]")
        self.assertEqual(text.replace("score_9", "").count("_"), 0)
        self.assertIn("blue hair", out)
        self.assertIn("score_9", out)
        self.assertIn("<lora:detail:0.6>", out)

    def test_natural_language_drops_sd_boilerplate(self):
        out = pf.finalize("gorgeous 1girl, solo, (smiling:1.2), masterpiece, 8k, trending on artstation", pf.NATURAL)
        self.assertEqual(out, "Gorgeous girl, smiling")

    def test_empty_prompt_is_unchanged(self):
        self.assertEqual(pf.finalize("", pf.ANIMA), "")


class PresetFallbackTests(unittest.TestCase):
    def test_unknown_preset_uses_standard(self):
        from random_prompt.one_button_presets import OneButtonPresets
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "obp_presets.json"
            user.write_text(json.dumps({"Standard": {"subject": "all", "insanitylevel": 5}}), encoding="utf-8")
            with mock.patch.object(OneButtonPresets, "OBP_FILE", user), \
                 mock.patch.object(OneButtonPresets, "DEFAULT_OBP_FILE", user):
                presets = OneButtonPresets()
                self.assertEqual(presets.get_obp_preset("Photography")["insanitylevel"], 5)
                self.assertEqual(presets.get_obp_preset(presets.CUSTOM_OBP)["subject"], "all")


if __name__ == "__main__":
    unittest.main()
