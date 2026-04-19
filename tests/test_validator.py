"""Tests for modules/validator.py"""
import json
import unittest
from datetime import date

import pandas as pd

from modules.validator import (
    AvailabilityRecord,
    StructuralError,
    validate_schema,
    validate_semantic,
    validate_structural,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _make_record(**kwargs) -> AvailabilityRecord:
    defaults = {"property_name": "Scotia Place"}
    defaults.update(kwargs)
    return AvailabilityRecord(**defaults)


def _make_inventory(names: list) -> pd.DataFrame:
    return pd.DataFrame({"Property Name": names})


# ── validate_structural ────────────────────────────────────────────────────────

class TestValidateStructural(unittest.TestCase):

    def test_valid_json(self):
        payload = {"listings": []}
        result = validate_structural(json.dumps(payload))
        self.assertEqual(result, payload)

    def test_strips_markdown_fences(self):
        raw = "```json\n{\"listings\": []}\n```"
        result = validate_structural(raw)
        self.assertEqual(result, {"listings": []})

    def test_strips_plain_fences(self):
        raw = "```\n{\"listings\": []}\n```"
        result = validate_structural(raw)
        self.assertEqual(result, {"listings": []})

    def test_raises_on_invalid_json(self):
        with self.assertRaises(StructuralError):
            validate_structural("not json at all")

    def test_raises_on_empty_string(self):
        with self.assertRaises(StructuralError):
            validate_structural("")


# ── validate_schema ────────────────────────────────────────────────────────────

class TestValidateSchema(unittest.TestCase):

    def _parsed(self, listings):
        return {"listings": listings}

    def test_valid_minimal_record(self):
        records = validate_schema(self._parsed([{"property_name": "Epcor Tower"}]))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].property_name, "Epcor Tower")

    def test_skips_invalid_record_logs_rest(self):
        listings = [
            {"property_name": "Good Building"},
            {"property_name": 12345},   # wrong type — will fail or coerce
            {},                          # missing required field
        ]
        records = validate_schema(self._parsed(listings))
        # At least the first valid record passes; bad ones are skipped
        names = [r.property_name for r in records]
        self.assertIn("Good Building", names)

    def test_full_record_with_all_fields(self):
        listing = {
            "property_name": "Scotia Place",
            "suite": "1200",
            "floor": 12,
            "size_sf": 4500.0,
            "headlease_sublease": "HL",
            "min_rent": 18.50,
            "max_rent": 22.00,
            "rent_type": "Net",
            "op_cost": 16.54,
            "op_cost_year": 2025,
            "availability": "Immediately",
            "occupancy_status": "Vacant",
            "confidence": "high",
        }
        records = validate_schema(self._parsed([listing]))
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.suite, "1200")
        self.assertEqual(rec.floor, 12)
        self.assertAlmostEqual(rec.size_sf, 4500.0)

    def test_confidence_defaults_to_medium(self):
        records = validate_schema(self._parsed([{"property_name": "X"}]))
        self.assertEqual(records[0].confidence, "medium")

    def test_empty_listings_returns_empty_list(self):
        self.assertEqual(validate_schema({"listings": []}), [])

    def test_missing_listings_key_returns_empty(self):
        self.assertEqual(validate_schema({}), [])


# ── validate_semantic ──────────────────────────────────────────────────────────

class TestValidateSemantic(unittest.TestCase):

    def _run(self, records, pdf_text, inv_names=None):
        inv_df = _make_inventory(inv_names or ["Scotia Place", "Epcor Tower"])
        return validate_semantic(records, pdf_text, inv_df)

    def test_passes_clean_records(self):
        rec = _make_record(
            property_name="Scotia Place",
            size_sf=2000.0,
            min_rent=18.50,
        )
        pdf = "Scotia Place offers 2000 sf of Class A office space."
        valid, issues = self._run([rec], pdf, ["Scotia Place"])
        self.assertIn(rec, valid)
        self.assertEqual(len(issues), 0)

    def test_empty_records_is_issue(self):
        _, issues = self._run([], "some text")
        rule_names = [i["rule"] for i in issues]
        self.assertIn("min_records", rule_names)

    def test_rent_out_of_range_flagged(self):
        rec = _make_record(property_name="Scotia Place", min_rent=150.0)
        pdf = "Scotia Place is located downtown."
        _, issues = self._run([rec], pdf, ["Scotia Place"])
        rules = [i["rule"] for i in issues]
        self.assertIn("rent_range", rules)

    def test_size_out_of_range_flagged(self):
        rec = _make_record(property_name="Scotia Place", size_sf=999_000.0)
        pdf = "Scotia Place is located downtown."
        _, issues = self._run([rec], pdf, ["Scotia Place"])
        rules = [i["rule"] for i in issues]
        self.assertIn("size_range", rules)

    def test_property_name_not_in_pdf_flagged(self):
        rec = _make_record(property_name="Phantom Tower")
        pdf = "Scotia Place is a great building."
        _, issues = self._run([rec], pdf, ["Scotia Place", "Phantom Tower"])
        rules = [i["rule"] for i in issues]
        self.assertIn("property_name_in_text", rules)

    def test_hallucinated_name_fails_fuzzy(self):
        rec = _make_record(property_name="Completely Made Up Xyz Tower")
        pdf = "Completely Made Up Xyz Tower is here."
        _, issues = self._run([rec], pdf, ["Scotia Place", "Epcor Tower"])
        rules = [i["rule"] for i in issues]
        self.assertIn("inventory_fuzzy_match", rules)

    def test_no_inventory_skips_fuzzy_check(self):
        rec = _make_record(property_name="Scotia Place", size_sf=1000.0, min_rent=18.0)
        pdf = "Scotia Place is a building."
        valid, issues = validate_semantic([rec], pdf, pd.DataFrame())
        # Without inventory, fuzzy check is skipped — record should pass
        self.assertIn(rec, valid)

    def test_pydantic_records_pass_through_correctly(self):
        """All records from validate_schema should be acceptable inputs."""
        parsed = {"listings": [{"property_name": "Epcor Tower", "size_sf": 5000.0}]}
        schema_records = validate_schema(parsed)
        pdf = "Epcor Tower is in downtown Edmonton."
        valid, issues = validate_semantic(schema_records, pdf, _make_inventory(["Epcor Tower"]))
        self.assertEqual(len(valid), 1)


if __name__ == "__main__":
    unittest.main()
