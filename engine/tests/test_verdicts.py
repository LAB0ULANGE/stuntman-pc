"""Remembered verification verdicts: a verdict is reused only for the same text,
form and environment."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ps2decomp import verdicts  # noqa: E402

REPOSITORY = os.path.join(os.path.dirname(__file__), "..", "..")


class TestVerdicts(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.path = os.path.join(self.folder, "verdicts.json")

    def test_same_text_form_and_environment_is_remembered(self):
        cache = verdicts.Verdicts(self.path, "env")
        cache.put("void F() {}", False, "verified")
        cache.save()
        again = verdicts.Verdicts(self.path, "env")
        self.assertEqual(again.get("void F() {}", False), "verified")

    def test_other_text_form_or_environment_is_not(self):
        cache = verdicts.Verdicts(self.path, "env")
        cache.put("void F() {}", False, "verified")
        cache.save()
        again = verdicts.Verdicts(self.path, "env")
        self.assertIsNone(again.get("void F() { }", False))
        self.assertIsNone(again.get("void F() {}", True))
        self.assertIsNone(verdicts.Verdicts(self.path, "other").get("void F() {}", False))

    def test_no_cache_ignores_but_still_records(self):
        cache = verdicts.Verdicts(self.path, "env")
        cache.put("t", False, "verified")
        cache.save()
        fresh = verdicts.Verdicts(self.path, "env", use=False)
        self.assertIsNone(fresh.get("t", False))
        fresh.put("t", False, "disagrees: memory")
        fresh.save()
        self.assertEqual(verdicts.Verdicts(self.path, "env").get("t", False), "disagrees: memory")

    def test_a_few_environments_are_kept(self):
        for index in range(verdicts.KEEP_ENVIRONMENTS + 2):
            cache = verdicts.Verdicts(self.path, "env%d" % index)
            cache.put("t", False, str(index))
            cache.save()
        last = verdicts.KEEP_ENVIRONMENTS + 1
        self.assertEqual(verdicts.Verdicts(self.path, "env%d" % last).get("t", False), str(last))
        self.assertEqual(verdicts.Verdicts(self.path, "env%d" % (last - 1)).get("t", False),
                         str(last - 1))
        self.assertIsNone(verdicts.Verdicts(self.path, "env0").get("t", False))

    def test_fingerprint_follows_inputs_not_text_only_modules(self):
        base = verdicts.fingerprint(REPOSITORY, (b"image",))
        self.assertEqual(base, verdicts.fingerprint(REPOSITORY, (b"image",)))
        self.assertNotEqual(base, verdicts.fingerprint(REPOSITORY, (b"other image",)))
        names = {os.path.basename(path) for path in verdicts._files(REPOSITORY)}
        self.assertIn("emu.py", names)
        self.assertIn("lifted.h", names)
        self.assertFalse(names & verdicts.TEXT_ONLY)

    def test_the_symbolic_deriver_is_not_part_of_the_environment(self):
        # Lift verification never runs it; a change to it must not cost a cold run.
        package = os.path.join(REPOSITORY, "engine", "ps2decomp")
        names = verdicts.modules(package)
        self.assertIn("liftverify.py", names)
        self.assertIn("testplan.py", names)
        self.assertFalse(names & {"paths.py", "symexec.py", "calls.py"})


if __name__ == "__main__":
    unittest.main()
