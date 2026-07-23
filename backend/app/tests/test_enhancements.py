import unittest
from app.services.clean_text import clean_extracted_text
from app.services.grounding_verifier import verify_and_clean_grounding
from app.services.simplify import _score_risk
from app.services.split_clauses import split_into_clauses


class TestEnhancements(unittest.TestCase):

    def test_grounding_verifier_detects_hallucination(self):
        source_clause = "The Tenant shall pay monthly rent on or before the 5th day of each month."
        hallucinated_output = "You must pay Rs. 50,000 rent by the 5th day of each month."
        cleaned, modified = verify_and_clean_grounding(source_clause, hallucinated_output)
        self.assertTrue(modified)
        self.assertNotIn("50,000", cleaned)

    def test_grounding_verifier_passes_grounded_text(self):
        source_clause = "The Tenant shall pay Rs. 20,000 security deposit."
        grounded_output = "You must pay Rs. 20,000 security deposit."
        cleaned, modified = verify_and_clean_grounding(source_clause, grounded_output)
        self.assertFalse(modified)
        self.assertEqual(cleaned, grounded_output)

    def test_risk_score_negation_awareness(self):
        negated_clause = "The security deposit is not subject to forfeiture under any normal circumstance."
        result = _score_risk(negated_clause)
        self.assertNotIn("forfeit", result["flags"])
        self.assertLess(result["risk_score"], 70)

    def test_clause_splitter_roman_numerals(self):
        text = "I. First clause text\nII. Second clause text\nIII. Third clause text"
        clauses = split_into_clauses(text)
        self.assertEqual(len(clauses), 3)

    def test_ocr_cleaner(self):
        ocr_text = "The Teant shall pay rent to Licensorr every month."
        cleaned = clean_extracted_text(ocr_text)
        self.assertIn("Tenant", cleaned)
        self.assertIn("Licensor", cleaned)


if __name__ == "__main__":
    unittest.main()
