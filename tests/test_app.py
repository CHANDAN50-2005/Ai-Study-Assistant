import sys
import unittest
import hashlib
from pathlib import Path

from streamlit.testing.v1 import AppTest
from study_engine import SourcePage, generate_multiple_choice_quiz, generate_quiz



PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
APP_PATH = PROJECT_ROOT / "app.py"


class AppSmokeTests(unittest.TestCase):
    sample = (
        "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
        "Chlorophyll absorbs sunlight in the leaves. "
        "The plant uses this energy to convert carbon dioxide and water into glucose and oxygen. "
        "Glucose stores energy that supports plant growth."
    )

    def test_empty_app_renders(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Build a better study session" in element.value for element in app.title))

    def test_pasted_material_opens_study_workspace(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        app.text_area(key="notes_input_0").set_value(
            "Photosynthesis converts light into chemical energy. Chlorophyll absorbs sunlight."
        ).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Your study workspace" in element.value for element in app.title))
        self.assertTrue(any("Sentences" in element.label for element in app.metric))

    def test_overview_question_renders_summary_answer(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        app.text_area(key="notes_input_0").set_value(self.sample).run()
        app.segmented_control(key="study_tool").set_value("Ask questions").run()
        fingerprint = hashlib.sha256(f"None:{self.sample}".encode("utf-8")).hexdigest()[:16]
        app.text_input(key=f"question_{fingerprint}").set_value("What is this PDF about?")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Document overview" in item.value for item in app.subheader))
        self.assertTrue(any("Overall, this document focuses" in item.value for item in app.markdown))

    def test_focus_question_and_unsupported_question_paths_render(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        app.text_area(key="notes_input_0").set_value(self.sample).run()
        app.segmented_control(key="study_tool").set_value("Ask questions").run()
        fingerprint = hashlib.sha256(f"None:{self.sample}".encode("utf-8")).hexdigest()[:16]
        app.text_input(key=f"question_{fingerprint}").set_value("What does chlorophyll absorb?")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Answer from your material" in item.value for item in app.subheader))
        self.assertTrue(any("According to your material:" in item.value for item in app.markdown))

        app.text_input(key=f"question_{fingerprint}").set_value("Who invented the telephone?")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(item.proto.title == "No clear match found" for item in app.info))

    def test_quiz_form_checks_and_scores_answers(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        app.text_area(key="notes_input_0").set_value(self.sample).run()
        app.segmented_control(key="study_tool").set_value("Practice quiz").run()
        fingerprint = hashlib.sha256(f"None:{self.sample}".encode("utf-8")).hexdigest()[:16]
        questions = generate_quiz([SourcePage(None, self.sample)], 5)
        self.assertTrue(questions)
        app.text_input(key=f"quiz_{fingerprint}_1").set_value(questions[0].answer)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(item.proto.title == "Quiz checked" for item in app.success))

    def test_explore_flashcards_render(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        app.text_area(key="notes_input_0").set_value(self.sample).run()
        app.segmented_control(key="study_tool").set_value("Explore").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Flashcards" in item.value for item in app.subheader))

    def test_explore_multiple_choice_renders(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        app.text_area(key="notes_input_0").set_value(self.sample).run()
        app.segmented_control(key="study_tool").set_value("Explore").run()
        app.segmented_control(key="explore_tool").set_value("Multiple choice").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Multiple-choice quiz" in item.value for item in app.subheader))

    def test_multiple_choice_form_scores_correct_responses(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        app.text_area(key="notes_input_0").set_value(self.sample).run()
        app.segmented_control(key="study_tool").set_value("Explore").run()
        app.segmented_control(key="explore_tool").set_value("Multiple choice").run()
        fingerprint = hashlib.sha256(f"None:{self.sample}".encode("utf-8")).hexdigest()[:16]
        questions = generate_multiple_choice_quiz([SourcePage(None, self.sample)], 5)
        for index, question in enumerate(questions, start=1):
            app.radio(key=f"choice_{fingerprint}_{index}").set_value(question.answer)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(item.proto.title == "Quiz checked" for item in app.success))
        self.assertTrue(any(f"{len(questions)} out of {len(questions)}" in item.value for item in app.success))

    def test_explore_keywords_render(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        app.text_area(key="notes_input_0").set_value(self.sample).run()
        app.segmented_control(key="study_tool").set_value("Explore").run()
        app.segmented_control(key="explore_tool").set_value("Topics & keywords").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Topics & keywords" in item.value for item in app.subheader))


if __name__ == "__main__":
    unittest.main()
