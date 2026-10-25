import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from study_engine import (  # noqa: E402
    DocumentError,
    SourcePage,
    answer_question,
    answers_match,
    clean_pdf_text,
    compose_overview,
    extract_pdf_pages,
    generate_quiz,
    is_document_overview_question,
    natural_answer_prefix,
    pages_from_text,
    split_passages,
    summarize,
)


SAMPLE = (
    "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
    "Chlorophyll absorbs sunlight in the leaves. "
    "The plant uses this energy to convert carbon dioxide and water into glucose and oxygen. "
    "Glucose stores energy that supports plant growth."
)


def make_text_pdf(text: str) -> bytes:
    """Build a tiny one-page PDF fixture using only the Python standard library."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")
    stream = b"BT /F1 12 Tf 72 720 Td (" + escaped + b") Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{object_number} 0 obj\n".encode("ascii"))
        data.extend(body)
        data.extend(b"\nendobj\n")
    xref_offset = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(data)


class TextProcessingTests(unittest.TestCase):
    def test_empty_input_has_no_pages_or_sentences(self):
        self.assertEqual(pages_from_text("  \n "), [])
        self.assertEqual(split_passages([]), [])
        self.assertEqual(summarize([], 3), [])
        self.assertFalse(answer_question([], "What is this about?").confident)
        self.assertEqual(generate_quiz([], 5), [])

    def test_sentence_split_protects_abbreviations_and_decimals(self):
        pages = pages_from_text("Use e.g. 3.14 as an example. Then calculate the result.")
        sentences = split_passages(pages)
        self.assertEqual(len(sentences), 2)
        self.assertIn("e.g.", sentences[0].text)
        self.assertIn("3.14", sentences[0].text)

    def test_pdf_text_cleaner_repairs_hyphenated_line_wrap(self):
        self.assertEqual(clean_pdf_text("photo-\nsynthesis converts light."), "photosynthesis converts light.")

    def test_pdf_text_cleaner_removes_index_terms_and_journal_footer(self):
        text = "Index Terms - UPI, fraud detection, machine learning.\nwww.ijcrt.org © 2025 IJCRT | Volume 13, Issue 4.\nFraud detection identifies suspicious transactions."
        cleaned = clean_pdf_text(text)
        self.assertEqual(cleaned, "Fraud detection identifies suspicious transactions.")

    def test_summary_returns_central_sentences_in_original_order(self):
        pages = pages_from_text(SAMPLE)
        summary = summarize(pages, 2)
        self.assertEqual(len(summary), 2)
        self.assertEqual([item.text for item in summary], sorted([item.text for item in summary], key=SAMPLE.index))
        self.assertTrue(all(item.text in SAMPLE for item in summary))

    def test_overview_phrasings_use_summary(self):
        for question in (
            "What is this PDF about?",
            "What does the research paper discuss?",
            "Give me an overview",
            "Summarize the main idea",
        ):
            self.assertTrue(is_document_overview_question(question), question)
        result = answer_question(pages_from_text(SAMPLE), "What is this PDF about?")
        self.assertTrue(result.confident)
        self.assertTrue(result.overview)
        self.assertEqual(len(result.evidence), 3)
        self.assertTrue(compose_overview(result.evidence).startswith("Overall, this document focuses"))

    def test_focused_question_returns_supporting_evidence(self):
        result = answer_question(pages_from_text(SAMPLE), "What does chlorophyll absorb?")
        self.assertTrue(result.confident)
        self.assertIn("Chlorophyll", result.answer)
        self.assertEqual(result.evidence[0].passage.page, None)

    def test_related_word_forms_match(self):
        result = answer_question(pages_from_text(SAMPLE), "How do plants convert light?")
        self.assertTrue(result.confident)
        self.assertTrue(result.evidence)

    def test_unrelated_question_does_not_invent_an_answer(self):
        result = answer_question(pages_from_text(SAMPLE), "Who invented the telephone?")
        self.assertFalse(result.confident)
        self.assertEqual(result.answer, "")

    def test_natural_answer_lead_in_matches_question_intent(self):
        self.assertEqual(natural_answer_prefix("Why does this happen?"), "Your material explains that:")
        self.assertEqual(natural_answer_prefix("How does this work?"), "Your material describes the process this way:")

    def test_quiz_questions_have_unique_answers_and_check_case_insensitively(self):
        questions = generate_quiz(pages_from_text(SAMPLE), 5)
        self.assertGreaterEqual(len(questions), 2)
        self.assertEqual(len({question.answer.lower() for question in questions}), len(questions))
        self.assertTrue(answers_match("  GLUCOSE! ", "glucose"))
        self.assertFalse(answers_match("oxygen", "glucose"))

    def test_pdf_extraction_preserves_page_number_and_text(self):
        pages = extract_pdf_pages(make_text_pdf("Photosynthesis converts light into energy."))
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].number, 1)
        result = answer_question(pages, "What does photosynthesis convert?")
        self.assertTrue(result.confident)
        self.assertEqual(result.evidence[0].passage.page, 1)

    def test_invalid_and_image_only_pdf_are_reported(self):
        with self.assertRaises(DocumentError):
            extract_pdf_pages(b"not a pdf")
        with self.assertRaises(DocumentError):
            extract_pdf_pages(make_text_pdf(""))


if __name__ == "__main__":
    unittest.main()
