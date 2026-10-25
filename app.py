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
    extract_pdf_pages,
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


def render_passage(passage: str, source_label: str, number: int | None = None) -> None:
    with st.container(border=True):
        if number is not None:
            st.caption(f"KEY IDEA {number:02d} · {source_label}")
        else:
            st.caption(source_label)
        st.write(passage)


st.session_state.setdefault("material_revision", 0)
st.session_state.setdefault("last_answer", None)
st.session_state.setdefault("last_quiz_result", None)

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
    st.title("Build a better study session", icon=":material/auto_stories:")
    st.write("Add your notes or a text-based PDF. Get a clear overview, ask focused questions, and practise key terms.")
    st.space("small")

    with st.container(border=True):
        st.subheader("Start with your material")
        st.write("Choose **Paste text** in the sidebar for notes, or **Upload PDF** for a paper or chapter.")
        st.caption("This project runs locally. It uses NLP retrieval and extractive summaries; it does not send your notes to an AI service.")

    first, second, third = st.columns(3)
    with first, st.container(border=True):
        st.markdown("### :material/summarize: Get the overview")
        st.write("Find central sentences from the supplied text.")
    with second, st.container(border=True):
        st.markdown("### :material/forum: Ask questions")
        st.write("Find relevant passages and see where they came from.")
    with third, st.container(border=True):
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
        options=["Summary", "Ask questions", "Practice quiz"],
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
                with st.container(border=True):
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
        st.caption("Answers are retrieved from your notes. This NLP mini-project does not invent or paraphrase facts.")

    else:
        st.subheader("Practise active recall")
        st.write("Fill in the missing key term, then check your answers.")
        question_count = st.slider("Number of questions", min_value=1, max_value=10, value=5)
        quiz = generate_quiz(pages, question_count)
        if not quiz:
            st.info("There is not enough text with suitable key terms to make a quiz yet.", title="Add more study material")
        else:
            with st.form(f"quiz_form_{fingerprint}"):
                for index, item in enumerate(quiz, start=1):
                    st.markdown(f"**{index}. {item.prompt}**")
                    st.text_input(
                        f"Your answer · question {index}",
                        key=f"quiz_{fingerprint}_{index}",
                        placeholder="Type the missing term",
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

