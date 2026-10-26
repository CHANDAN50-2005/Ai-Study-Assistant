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
    extract_key_terms,
    generate_flashcards,
    generate_multiple_choice_quiz,
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

    def test_hyphenated_pdf_terms_match_questions_written_with_spaces(self):
        pages = pages_from_text(
            "Column-family stores allow you to store data with keys mapped to values."
        )
        result = answer_question(pages, "What do column family stores allow you to store?")
        self.assertTrue(result.confident)
        self.assertIn("keys mapped to values", result.answer)

    def test_lsa_retrieves_relevant_source_for_paraphrased_questions(self):
        pages = pages_from_text(
            "Photosynthesis converts sunlight into chemical energy in green plants. "
            "Chlorophyll is the green pigment that captures light for photosynthesis. "
            "Plants use this process to transform solar energy into stored food. "
            "Sunlight is absorbed by chlorophyll in plant leaves."
        )
        result = answer_question(pages, "What does the green substance trap?")
        self.assertTrue(result.confident)
        self.assertIn("Chlorophyll", result.answer)
        process = answer_question(pages, "What does the plant turn sunlight into?")
        self.assertTrue(process.confident)
        self.assertIn("converts sunlight into chemical energy", process.answer)

    def test_code_examples_do_not_become_answers_or_quiz_questions(self):
        pages = pages_from_text(
            "Insert into Column family by Cassandra client • "
            "SET Customer['mfwoler']['city']='Boston'; • "
            "Column-family stores, such as Cassandra and HBase, allow you to store data with keys mapped to values."
        )
        questions = generate_quiz(pages, 5)
        self.assertTrue(questions)
        self.assertTrue(all("mfwoler" not in question.prompt for question in questions))
        result = answer_question(pages, "What is the city of mfwoler?")
        self.assertFalse(result.confident)
        definition = answer_question(pages, "What do column family stores allow?")
        self.assertTrue(definition.confident)
        self.assertIn("keys mapped to values", definition.answer)

    def test_database_pdf_example_creates_a_concept_question_not_a_random_cloze(self):
        pages = pages_from_text(
            "Insert into Column family by Cassandra client • "
            "SET Customer['mfwoler']['city']='Boston'; • "
            "We can read the data back using the GET command. • "
            "Column-family Stores • Column-family stores, such as Cassandra, HBase, ________, and Amazon SimpleDB allow you to store data with keys mapped to values and the values grouped into multiple column families."
        )
        questions = generate_quiz(pages, 5)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].answer.casefold(), "column-family stores")
        self.assertIn("Which concept is described", questions[0].prompt)
        self.assertNotIn("mfwoler", questions[0].prompt)

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

    def test_flashcards_connect_a_key_term_to_its_source_sentence(self):
        cards = generate_flashcards(pages_from_text(SAMPLE), 3)
        self.assertTrue(cards)
        for card in cards:
            self.assertIn(card.front, card.back)

    def test_multiple_choice_questions_have_unique_choices_and_correct_answer(self):
        questions = generate_multiple_choice_quiz(pages_from_text(SAMPLE), 3)
        self.assertTrue(questions)
        for question in questions:
            self.assertIn(question.answer, question.choices)
            self.assertEqual(len(question.choices), len(set(question.choices)))

    def test_multiple_choice_needs_multiple_distinct_concepts(self):
        pages = pages_from_text("Photosynthesis is the process by which plants convert light into chemical energy.")
        self.assertEqual(generate_multiple_choice_quiz(pages, 5), [])

    def test_key_terms_rank_subject_words_from_material(self):
        terms = extract_key_terms(pages_from_text(SAMPLE), 8)
        labels = {item.term.casefold() for item in terms}
        self.assertTrue(labels & {"photosynthesis", "chlorophyll", "glucose", "sunlight"})
        self.assertTrue(all(item.frequency >= 1 for item in terms))

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
