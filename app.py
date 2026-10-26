from __future__ import annotations

import hashlib
import re
from pathlib import Path

import streamlit as st

from study_engine import (
    AnswerResult,
    DocumentError,
    SourcePage,
    answer_question,
    answers_match,
    compose_overview,
    extract_key_terms,
    extract_pdf_pages,
    generate_flashcards,
    generate_multiple_choice_quiz,
    generate_quiz,
    natural_answer_prefix,
    pages_from_text,
    split_passages,
    summarize,
)


st.set_page_config(
    page_title="NLP Study Assistant",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html("""
<style>
:root {
  --study-accent: #ff5b1f;
  --study-accent-soft: rgba(255, 91, 31, .14);
  --study-border: rgba(255, 255, 255, .09);
  --study-muted: #a0a3a7;
}

.stApp {
  background:
    radial-gradient(ellipse at 78% 8%, rgba(255, 91, 31, .075), transparent 34%),
    radial-gradient(ellipse at 12% 92%, rgba(255, 255, 255, .035), transparent 32%),
    #101112;
}
[data-testid="stHeader"] { background: rgba(16, 17, 18, .72); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #17191a 0%, #121415 100%);
  border-right: 1px solid var(--study-border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label { color: #d0d2d4; }
.block-container { padding-top: 2.3rem; padding-bottom: 3rem; }
h1, h2, h3 { letter-spacing: -.035em; }
h1 { font-weight: 750; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--study-muted); }

.st-key-welcome-hero,
.st-key-workspace-header,
.st-key-answer-panel,
.st-key-feature-summary,
.st-key-feature-questions,
.st-key-feature-quiz,
.st-key-summary-card-1,
.st-key-summary-card-2,
.st-key-summary-card-3,
.st-key-summary-card-4,
.st-key-summary-card-5,
.st-key-summary-card-6,
.st-key-summary-card-7,
.st-key-summary-card-8,
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(39, 41, 43, .92), rgba(24, 26, 28, .88));
  border: 1px solid var(--study-border);
  border-radius: 18px;
  box-shadow: 0 14px 36px rgba(0, 0, 0, .20), inset 0 1px rgba(255, 255, 255, .035);
}
.st-key-welcome-hero, .st-key-workspace-header { padding: 1.35rem 1.55rem; }
.st-key-feature-summary, .st-key-feature-questions, .st-key-feature-quiz,
.st-key-summary-card-1, .st-key-summary-card-2, .st-key-summary-card-3,
.st-key-summary-card-4, .st-key-summary-card-5, .st-key-summary-card-6,
.st-key-summary-card-7, .st-key-summary-card-8 { padding: .9rem 1.05rem; }
[data-testid="stMetric"] { padding: .85rem 1rem; }
[data-testid="stMetricLabel"] { color: var(--study-muted); }
[data-testid="stMetricValue"] { color: #fff; }

.stButton > button, .stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
  border-radius: 11px;
  border: 1px solid rgba(255, 91, 31, .48);
  transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
}
.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, #ff6b2c, #e94712);
  color: #fff;
  box-shadow: 0 6px 18px rgba(255, 76, 20, .2);
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 22px rgba(255, 76, 20, .18);
  border-color: var(--study-accent);
}
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
  background: var(--study-accent-soft);
  color: #ff875a;
  border-color: rgba(255, 91, 31, .45);
}
input, textarea, [data-baseweb="select"] > div { border-radius: 10px !important; }
[data-testid="stExpander"] {
  border-color: var(--study-border);
  border-radius: 13px;
  background: rgba(25, 27, 29, .58);
}

@keyframes study-assistant-enter {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

.st-key-welcome-hero,
.st-key-workspace-header,
.st-key-answer-panel {
  animation: study-assistant-enter 480ms cubic-bezier(.2, .7, .25, 1) both;
}

.st-key-feature-summary,
.st-key-feature-questions,
.st-key-feature-quiz {
  animation: study-assistant-enter 520ms cubic-bezier(.2, .7, .25, 1) both;
  transition: transform 180ms ease, box-shadow 180ms ease;
}

.st-key-feature-summary { animation-delay: 60ms; }
.st-key-feature-questions { animation-delay: 130ms; }
.st-key-feature-quiz { animation-delay: 200ms; }

.st-key-feature-summary:hover,
.st-key-feature-questions:hover,
.st-key-feature-quiz:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 30px rgba(0, 0, 0, .32), 0 0 22px rgba(255, 91, 31, .08);
}

.st-key-summary-card-1,
.st-key-summary-card-2,
.st-key-summary-card-3,
.st-key-summary-card-4,
.st-key-summary-card-5,
.st-key-summary-card-6,
.st-key-summary-card-7,
.st-key-summary-card-8 {
  animation: study-assistant-enter 420ms cubic-bezier(.2, .7, .25, 1) both;
}

@media (prefers-reduced-motion: reduce) {
  .st-key-welcome-hero,
  .st-key-workspace-header,
  .st-key-answer-panel,
  .st-key-feature-summary,
  .st-key-feature-questions,
  .st-key-feature-quiz,
  .st-key-summary-card-1,
  .st-key-summary-card-2,
  .st-key-summary-card-3,
  .st-key-summary-card-4,
  .st-key-summary-card-5,
  .st-key-summary-card-6,
  .st-key-summary-card-7,
  .st-key-summary-card-8 {
    animation: none !important;
    transition: none !important;
  }
}
</style>
""")


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def cached_pdf_pages(pdf_bytes: bytes) -> list[SourcePage]:
    return extract_pdf_pages(pdf_bytes)


def document_fingerprint(pages: list[SourcePage]) -> str:
    contents = "\n".join(f"{page.number}:{page.text}" for page in pages)
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()[:16]


def clear_material() -> None:
    st.session_state["material_revision"] = st.session_state.get("material_revision", 0) + 1
    st.session_state["last_answer"] = None
    st.session_state["last_quiz_result"] = None
    st.session_state["last_choice_result"] = None
    st.session_state["flashcard_index"] = 0
    st.session_state["flashcard_revealed"] = False


def move_flashcard(delta: int, total: int) -> None:
    current = st.session_state.get("flashcard_index", 0)
    st.session_state["flashcard_index"] = min(max(current + delta, 0), total - 1)
    st.session_state["flashcard_revealed"] = False


def render_passage(passage: str, source_label: str, number: int | None = None) -> None:
    key = f"summary-card-{number}" if number is not None else None
    with st.container(border=True, key=key):
        if number is not None:
            st.caption(f"KEY IDEA {number:02d} · {source_label}")
        else:
            st.caption(source_label)
        st.write(passage)


st.session_state.setdefault("material_revision", 0)
st.session_state.setdefault("last_answer", None)
st.session_state.setdefault("last_quiz_result", None)
st.session_state.setdefault("last_choice_result", None)
st.session_state.setdefault("flashcard_index", 0)
st.session_state.setdefault("flashcard_revealed", False)

with st.sidebar:
    st.title("Study assistant", icon=":material/school:")
    st.caption("Turn class notes and papers into focused revision sessions.")
    st.space("small")

    source_mode = st.segmented_control(
        "Add study material",
        options=["Paste text", "Upload PDF"],
        default="Paste text",
        key="source_mode",
        width="stretch",
    )
    revision = st.session_state["material_revision"]
    document_name = "Pasted study notes"
    load_error: str | None = None

    if source_mode == "Paste text":
        pasted_text = st.text_area(
            "Your notes",
            placeholder="Paste lecture notes, textbook passages, or a study guide...",
            height=300,
            key=f"notes_input_{revision}",
        )
        pages = pages_from_text(pasted_text)
    else:
        uploaded_pdf = st.file_uploader(
            "Choose a text-based PDF",
            type=["pdf"],
            max_upload_size=20,
            key=f"pdf_input_{revision}",
            help="Up to 20 MB. Scanned image-only PDFs need OCR, which is not included.",
        )
        pages = []
        if uploaded_pdf is not None:
            document_name = uploaded_pdf.name
            try:
                with st.spinner("Reading PDF pages..."):
                    pages = cached_pdf_pages(uploaded_pdf.getvalue())
            except DocumentError as exc:
                load_error = str(exc)

    if pages:
        st.success("Study material ready", icon=":material/check_circle:")
    elif load_error:
        st.error(load_error, title="Could not read this PDF", icon=":material/error:")
    else:
        st.caption("Your material stays in this local study session.")

    if pages or load_error:
        st.button(
            "Clear material",
            icon=":material/delete_outline:",
            width="stretch",
            on_click=clear_material,
        )


if not pages:
    with st.container(key="welcome-hero"):
        st.title("Build a better study session", icon=":material/auto_stories:")
        st.write("Add your notes or a text-based PDF. Get a clear overview, ask focused questions, and practise key terms.")
        st.space("small")
        st.subheader("Start with your material")
        st.write("Choose **Paste text** in the sidebar for notes, or **Upload PDF** for a paper or chapter.")
        st.caption("This project runs locally. It fits an LSA model to your material and keeps answers grounded in extracted passages; it does not send your notes to an AI service.")

    first, second, third = st.columns(3)
    with first:
        with st.container(border=True, key="feature-summary"):
            st.markdown("### :material/summarize: Get the overview")
            st.write("Find central sentences from the supplied text.")
    with second:
        with st.container(border=True, key="feature-questions"):
            st.markdown("### :material/forum: Ask questions")
            st.write("Find relevant passages and see where they came from.")
    with third:
        with st.container(border=True, key="feature-quiz"):
            st.markdown("### :material/quiz: Practise recall")
            st.write("Answer fill-in-the-blank questions and check your score.")
else:
    passages = split_passages(pages)
    all_text = " ".join(page.text for page in pages)
    word_count = len(re.findall(r"\b\w+\b", all_text))
    reading_minutes = max(1, round(word_count / 220))
    fingerprint = document_fingerprint(pages)

    if st.session_state.get("active_document") != fingerprint:
        st.session_state["active_document"] = fingerprint
        st.session_state["last_answer"] = None
        st.session_state["last_quiz_result"] = None
        st.session_state["last_choice_result"] = None
        st.session_state["flashcard_index"] = 0
        st.session_state["flashcard_revealed"] = False

    with st.container(key="workspace-header"):
        st.title("Your study workspace", icon=":material/auto_stories:")
        st.caption(f"Working from **{document_name}**")
        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Words", f"{word_count:,}")
        metric_two.metric("Sentences", f"{len(passages):,}")
        metric_three.metric("Reading time", f"{reading_minutes} min")

    if source_mode == "Upload PDF":
        with st.expander("Preview extracted PDF text"):
            preview = "\n\n".join(
                f"Page {page.number}\n{page.text}" for page in pages
            )
            st.text(preview[:10000] + ("\n\n[Preview shortened]" if len(preview) > 10000 else ""))

    active_tool = st.segmented_control(
        "Choose a study tool",
        options=["Summary", "Ask questions", "Practice quiz", "Explore"],
        default="Summary",
        key="study_tool",
    )

    if active_tool == "Summary":
        st.subheader("Key ideas")
        st.write("Central sentences from your material, kept in their original order.")
        max_summary = min(8, max(1, len(passages)))
        summary_count = st.slider(
            "Number of key sentences",
            min_value=1,
            max_value=max_summary,
            value=min(3, max_summary),
        )
        with st.spinner("Finding the central ideas..."):
            summary = summarize(pages, summary_count)
        for number, item in enumerate(summary, start=1):
            render_passage(item.text, f"Page {item.page}" if item.page else "Your notes", number)
        summary_text = "\n\n".join(
            f"{item.text} ({item.source_label})" if item.page else item.text for item in summary
        )
        st.download_button(
            "Download summary",
            data=summary_text,
            file_name=f"{Path(document_name).stem or 'study'}-summary.txt",
            mime="text/plain",
            icon=":material/download:",
        )

    elif active_tool == "Ask questions":
        st.subheader("Ask about your material")
        st.write("Try a question about a definition, process, cause, or the overall topic.")
        with st.form(f"question_form_{fingerprint}", border=False):
            question = st.text_input(
                "Your question",
                placeholder="For example: What is this paper about?",
                key=f"question_{fingerprint}",
            )
            submitted = st.form_submit_button("Find an answer", type="primary", icon=":material/search:")

        if submitted:
            if question.strip():
                with st.spinner("Looking for relevant passages..."):
                    answer = answer_question(pages, question)
                st.session_state["last_answer"] = (fingerprint, question.strip(), answer)
            else:
                st.warning("Enter a question first.", title="Question is empty")

        saved_answer = st.session_state.get("last_answer")
        if saved_answer and saved_answer[0] == fingerprint and saved_answer[1] == question.strip():
            answer_result: AnswerResult = saved_answer[2]
            if answer_result.overview and answer_result.confident:
                st.subheader("Document overview")
                with st.container(border=True):
                    st.write(compose_overview(answer_result.evidence))
                with st.expander("Show source sentences"):
                    for evidence in answer_result.evidence:
                        st.markdown(f"**{evidence.passage.source_label}**")
                        st.write(evidence.passage.text)
            elif answer_result.confident:
                st.subheader("Answer from your material")
                with st.container(border=True, key="answer-panel"):
                    st.write(f"{natural_answer_prefix(question)} {answer_result.answer}")
                with st.expander("Supporting passages"):
                    for evidence in answer_result.evidence:
                        st.markdown(f"**{evidence.passage.source_label} · relevance {evidence.score:.2f}**")
                        st.write(evidence.passage.text)
            else:
                st.info(
                    "I could not find a clear answer in this material. Try using a key term from your notes, or ask for an overview.",
                    title="No clear match found",
                    icon=":material/manage_search:",
                )
                if answer_result.evidence:
                    with st.expander("Possible related passages"):
                        for evidence in answer_result.evidence:
                            st.markdown(f"**{evidence.passage.source_label}**")
                            st.write(evidence.passage.text)
        st.caption("TF-IDF and document-trained LSA rank related passages. Answers stay grounded in your notes; this app does not generate unsupported facts.")

    elif active_tool == "Practice quiz":
        st.subheader("Practise active recall")
        st.write("Read each description and enter the concept it describes.")
        question_count = st.slider("Number of questions", min_value=1, max_value=10, value=5)
        quiz = generate_quiz(pages, question_count)
        if not quiz:
            st.info("I could not find enough clear definitions or concept statements to build a useful quiz. Try a more explanatory section of your notes.", title="No quiz-ready facts found")
        else:
            st.caption("Each question asks you to identify the concept described by a statement in your material.")
            with st.form(f"quiz_form_{fingerprint}"):
                for index, item in enumerate(quiz, start=1):
                    st.markdown(f"**{index}. {item.prompt}**")
                    st.text_input(
                        f"Your answer · question {index}",
                        key=f"quiz_{fingerprint}_{index}",
                        placeholder="Type the concept",
                    )
                checked = st.form_submit_button("Check my answers", type="primary", icon=":material/task_alt:")

            if checked:
                answers = [st.session_state.get(f"quiz_{fingerprint}_{index}", "") for index in range(1, len(quiz) + 1)]
                results = [answers_match(user, item.answer) for user, item in zip(answers, quiz)]
                st.session_state["last_quiz_result"] = (fingerprint, results)
            saved_quiz = st.session_state.get("last_quiz_result")
            if saved_quiz and saved_quiz[0] == fingerprint:
                results = saved_quiz[1]
                correct = sum(results)
                st.success(f"You got {correct} out of {len(quiz)} correct.", title="Quiz checked")
                with st.expander("Review answers", expanded=True):
                    for index, (item, is_correct) in enumerate(zip(quiz, results), start=1):
                        label = "Correct" if is_correct else f"Answer: {item.answer}"
                        st.markdown(f"**{index}. {label}** · {f'Page {item.page}' if item.page else 'Your notes'}")


    else:
        explore_tool = st.segmented_control(
            "Explore your material",
            options=["Flashcards", "Multiple choice", "Topics & keywords"],
            default="Flashcards",
            key="explore_tool",
            width="stretch",
        )

        if explore_tool == "Flashcards":
            st.subheader("Flashcards")
            st.write("Recall a key term, then reveal the sentence that explains it.")
            card_count = st.slider("Number of cards", min_value=1, max_value=10, value=5, key="flashcard_count")
            cards = generate_flashcards(pages, card_count)
            if not cards:
                st.info("There is not enough text with suitable key terms to make flashcards yet.", title="Add more study material")
            else:
                st.session_state["flashcard_index"] = min(st.session_state["flashcard_index"], len(cards) - 1)
                card_index = st.session_state["flashcard_index"]
                card = cards[card_index]
                st.progress((card_index + 1) / len(cards), text=f"Card {card_index + 1} of {len(cards)}")
                with st.container(border=True, key="flashcard-panel"):
                    st.caption("KEY TERM · FRONT")
                    st.markdown(f"## {card.front}")
                    if st.session_state["flashcard_revealed"]:
                        st.divider()
                        st.caption("FROM YOUR MATERIAL · BACK")
                        st.write(card.back)
                        st.caption(f"Page {card.page}" if card.page else "Your notes")
                st.toggle("Reveal the explanation", key="flashcard_revealed")
                previous, _, next_card = st.columns([1, 3, 1])
                previous.button(
                    "← Previous", disabled=card_index == 0, width="stretch",
                    on_click=move_flashcard, args=(-1, len(cards)),
                )
                next_card.button(
                    "Next →", disabled=card_index >= len(cards) - 1, width="stretch",
                    on_click=move_flashcard, args=(1, len(cards)),
                )

        elif explore_tool == "Multiple choice":
            st.subheader("Multiple-choice quiz")
            st.write("Choose the term that completes each sentence from your material.")
            choice_count = st.slider("Number of questions", min_value=1, max_value=10, value=5, key="choice_count")
            questions = generate_multiple_choice_quiz(pages, choice_count)
            if not questions:
                st.info("I need at least two clear concepts in the material to make a fair multiple-choice quiz.", title="Not enough distinct concepts")
            else:
                with st.form(f"choice_form_{fingerprint}"):
                    for index, item in enumerate(questions, start=1):
                        st.markdown(f"**{index}. {item.prompt}**")
                        st.radio(
                            f"Choose an answer · question {index}",
                            options=item.choices,
                            index=None,
                            key=f"choice_{fingerprint}_{index}",
                        )
                    checked = st.form_submit_button("Check my answers", type="primary", icon=":material/task_alt:")

                if checked:
                    answers = [st.session_state.get(f"choice_{fingerprint}_{index}", "") for index in range(1, len(questions) + 1)]
                    results = [answers_match(user, item.answer) for user, item in zip(answers, questions)]
                    st.session_state["last_choice_result"] = (fingerprint, results)
                saved_choices = st.session_state.get("last_choice_result")
                if saved_choices and saved_choices[0] == fingerprint:
                    results = saved_choices[1]
                    correct = sum(results)
                    st.success(f"You got {correct} out of {len(questions)} correct.", title="Quiz checked")
                    with st.expander("Review answers", expanded=True):
                        for index, (item, is_correct) in enumerate(zip(questions, results), start=1):
                            label = "Correct" if is_correct else f"Answer: {item.answer}"
                            st.markdown(f"**{index}. {label}** · {f'Page {item.page}' if item.page else 'Your notes'}")

        else:
            st.subheader("Topics & keywords")
            st.write("Explore the important terms ranked from your material using TF-IDF.")
            term_count = st.slider("Keywords to show", min_value=5, max_value=20, value=12, key="term_count")
            terms = extract_key_terms(pages, term_count)
            if not terms:
                st.info("I could not identify enough recurring terms. Try adding more study material.", title="No keywords found")
            else:
                columns = st.columns(3)
                for index, item in enumerate(terms):
                    with columns[index % 3]:
                        with st.container(border=True):
                            st.markdown(f"### {item.term.title()}")
                            st.caption(f"Appears at least {item.frequency} time{'s' if item.frequency != 1 else ''} · relevance {item.score:.2f}")
