"""NLP utilities for the NLP Study Assistant mini-project."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


QUESTION_FILLER_WORDS = {
    "according", "answer", "article", "book", "could", "describe", "document",
    "explain", "file", "give", "information", "mainly", "material", "mean",
    "meaning", "notes", "note", "paper", "pdf", "please", "provide", "question",
    "reading", "report", "research", "show", "study", "tell", "text", "define",
}
QUIZ_EXCLUDED_WORDS = {
    "abstract", "analysis", "author", "based", "conclusion", "figure", "following",
    "however", "introduction", "method", "methods", "paper", "present", "result",
    "results", "section", "shown", "study", "table", "therefore", "using",
}


class DocumentError(ValueError):
    """A document could not be read or did not contain usable text."""


@dataclass(frozen=True)
class SourcePage:
    number: int | None
    text: str


@dataclass(frozen=True)
class Passage:
    text: str
    page: int | None = None

    @property
    def source_label(self) -> str:
        return f"Page {self.page}" if self.page is not None else "Your notes"


@dataclass(frozen=True)
class ScoredPassage:
    passage: Passage
    score: float


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    evidence: tuple[ScoredPassage, ...]
    confident: bool
    overview: bool = False


@dataclass(frozen=True)
class QuizQuestion:
    prompt: str
    answer: str
    page: int | None = None


def pages_from_text(text: str) -> list[SourcePage]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [SourcePage(None, cleaned)] if cleaned else []


def clean_pdf_text(text: str) -> str:
    """Repair common line-wrap artifacts and discard obvious page furniture."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines() or [text]:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lowered = line.casefold()
        if re.match(r"^(?:index\s+terms?|keywords?)\b", lowered):
            continue
        if re.match(r"^(?:fig(?:ure)?|table)\s*\d", lowered):
            continue
        if re.match(r"^(?:https?://|www\.)", lowered):
            continue
        if any(marker in lowered for marker in (
            "www.ijcrt.org", "international journal of creative research thoughts",
            "copyright", "issn:", "© 20",
        )):
            continue
        if re.search(r"\bvolume\s+\d+\s*,?\s*issue\s+\d+\b", lowered):
            continue
        cleaned_lines.append(line)
    return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()


def extract_pdf_pages(pdf_bytes: bytes) -> list[SourcePage]:
    """Extract selectable text per page, retaining page numbers for citations."""
    if not pdf_bytes:
        raise DocumentError("The selected PDF is empty.")
    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:
                raise DocumentError("This PDF is password-protected. Remove the password and try again.") from exc
            if not unlocked:
                raise DocumentError("This PDF is password-protected. Remove the password and try again.")

        pages: list[SourcePage] = []
        for number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text(
                extraction_mode="layout",
                layout_mode_space_vertically=False,
            ) or ""
            page_text = clean_pdf_text(raw_text)
            if page_text:
                pages.append(SourcePage(number, page_text))
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("The PDF could not be read. It may be damaged or use an unsupported format.") from exc

    if not pages:
        raise DocumentError("No selectable text was found. This may be a scanned PDF; OCR is not included yet.")
    return pages


def split_sentences(text: str) -> list[str]:
    """Split ordinary prose into sentences while protecting common abbreviations."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    protected: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        marker = f"ABBREVIATIONTOKEN{len(protected)}X"
        protected[marker] = match.group(0)
        return marker

    text = re.sub(r"\b(?:e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|fig|eq|no)\.", protect, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\.(?=\d)", "DECIMALPOINTTOKEN", text)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    restored = []
    for sentence in sentences:
        for marker, original in protected.items():
            sentence = sentence.replace(marker, original)
        sentence = sentence.replace("DECIMALPOINTTOKEN", ".")
        restored.append(sentence)
    return restored


def split_passages(pages: list[SourcePage]) -> list[Passage]:
    passages: list[Passage] = []
    for page in pages:
        passages.extend(Passage(sentence, page.number) for sentence in split_sentences(page.text))
    return passages


def normalize_terms(text: str) -> list[str]:
    """Drop question framing and apply small, transparent suffix normalization."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())
    normalized: list[str] = []
    for token in tokens:
        if token in ENGLISH_STOP_WORDS or token in QUESTION_FILLER_WORDS:
            continue
        if token.endswith("ies") and len(token) > 5:
            token = token[:-3] + "y"
        elif token.endswith(("ches", "shes", "sses", "xes", "zes")) and len(token) > 5:
            token = token[:-2]
        elif token.endswith("ing") and len(token) > 6:
            token = token[:-3]
            if len(token) > 2 and token[-1] == token[-2]:
                token = token[:-1]
        elif token.endswith("ed") and len(token) > 5:
            token = token[:-2]
            if token.endswith("i"):
                token = token[:-1] + "y"
            elif len(token) > 2 and token[-1] == token[-2]:
                token = token[:-1]
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            token = token[:-1]
        normalized.append(token)
    return normalized


def is_document_overview_question(question: str) -> bool:
    cleaned = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    patterns = (
        r"\bwhat\s+(?:is|are)\s+(?:(?:this|the|it)\s+)?(?:(?:research\s+)?(?:pdf|document|paper|article|file|book|report|reading)\s+)?(?:mainly\s+)?about\b",
        r"\bwhat\s+(?:does|do)\s+(?:(?:this|the|it)\s+)?(?:(?:research\s+)?(?:pdf|document|paper|article|file|book|report|reading)\s+)?(?:cover|discuss|explain)\b",
        r"\b(?:summari[sz]e|overview|main idea|central theme|main topic)\b",
        r"\bwhat\s+is\s+(?:the\s+)?(?:main idea|main topic|central theme|purpose)\b",
    )
    return any(re.search(pattern, cleaned) for pattern in patterns)


def _centrality_scores(passages: list[Passage]) -> tuple[list[float], list[list[float]]]:
    if not passages:
        return [], []
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(
        [passage.text for passage in passages]
    )
    similarities = cosine_similarity(matrix)
    if len(passages) == 1:
        return [1.0], similarities.tolist()
    scores = (similarities.sum(axis=1) - similarities.diagonal()) / (len(passages) - 1)
    return scores.tolist(), similarities.tolist()


def summarize(pages: list[SourcePage], sentence_count: int = 3) -> list[Passage]:
    passages = split_passages(pages)
    if sentence_count <= 0 or not passages:
        return []
    if len(passages) <= sentence_count:
        return passages
    try:
        centrality, similarities = _centrality_scores(passages)
    except ValueError:
        return passages[:sentence_count]

    selected: list[int] = []
    candidates = set(range(len(passages)))
    while candidates and len(selected) < sentence_count:
        def diverse_score(index: int) -> float:
            redundancy = max((similarities[index][other] for other in selected), default=0.0)
            return centrality[index] - 0.22 * redundancy

        best = max(candidates, key=diverse_score)
        selected.append(best)
        candidates.remove(best)
    return [passages[index] for index in sorted(selected)]


def answer_question(pages: list[SourcePage], question: str, max_evidence: int = 3) -> AnswerResult:
    passages = split_passages(pages)
    if not passages or not question.strip():
        return AnswerResult("", (), False)

    if is_document_overview_question(question):
        overview = tuple(ScoredPassage(passage, 1.0) for passage in summarize(pages, 3))
        return AnswerResult("Overview from your material", overview, bool(overview), overview=True)

    question_terms = set(normalize_terms(question))
    if not question_terms:
        return AnswerResult("", (), False)
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=normalize_terms,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        vectors = vectorizer.fit_transform([question, *[passage.text for passage in passages]])
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).ravel()
    except ValueError:
        return AnswerResult("", (), False)

    scored: list[ScoredPassage] = []
    overlaps: list[int] = []
    for passage, similarity in zip(passages, similarities):
        sentence_terms = set(normalize_terms(passage.text))
        overlap = len(question_terms & sentence_terms)
        overlaps.append(overlap)
        if overlap:
            coverage = overlap / len(question_terms)
            score = 0.78 * float(similarity) + 0.22 * coverage
            scored.append(ScoredPassage(passage, score))
    scored.sort(key=lambda item: item.score, reverse=True)
    evidence = tuple(scored[:max(1, max_evidence)])
    confident = bool(evidence and evidence[0].score >= 0.12)
    answer = evidence[0].passage.text if confident else ""
    return AnswerResult(answer, evidence, confident)


def natural_answer_prefix(question: str) -> str:
    """Choose a short, grounded lead-in that fits the question type."""
    words = re.findall(r"[a-z]+", question.casefold())
    if "why" in words:
        return "Your material explains that:"
    if "how" in words:
        return "Your material describes the process this way:"
    if "who" in words:
        return "Your material identifies:"
    if "when" in words:
        return "Your material gives this timing detail:"
    if "where" in words:
        return "Your material locates it here:"
    return "According to your material:"


def compose_overview(passages: tuple[ScoredPassage, ...]) -> str:
    """Present extracted key sentences as a short, readable document overview."""
    sentences = [item.passage.text.strip() for item in passages if item.passage.text.strip()]
    if not sentences:
        return "I could not find enough readable text to summarize this document."
    return "Overall, this document focuses on the following ideas: " + " ".join(sentences)


def _answer_key(answer: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", answer.casefold()))


def answers_match(user_answer: str, expected_answer: str) -> bool:
    return bool(_answer_key(user_answer)) and _answer_key(user_answer) == _answer_key(expected_answer)


def generate_quiz(pages: list[SourcePage], count: int = 5) -> list[QuizQuestion]:
    passages = split_passages(pages)
    if count <= 0 or not passages:
        return []
    try:
        centrality, _ = _centrality_scores(passages)
    except ValueError:
        centrality = [1.0] * len(passages)
    normalized_passages = [normalize_terms(passage.text) for passage in passages]
    frequencies = Counter(term for terms in normalized_passages for term in terms)
    ordered = sorted(range(len(passages)), key=lambda index: centrality[index], reverse=True)
    questions: list[QuizQuestion] = []
    used_answers: set[str] = set()

    for index in ordered:
        passage = passages[index]
        words = re.findall(r"\b[A-Za-z][A-Za-z'-]{4,}\b", passage.text)
        candidates = [
            word for word in words
            if word.lower() not in ENGLISH_STOP_WORDS
            and word.lower() not in QUIZ_EXCLUDED_WORDS
            and normalize_terms(word)
            and normalize_terms(word)[0] not in used_answers
        ]
        if not candidates:
            continue
        answer = max(
            candidates,
            key=lambda word: (1 / (1 + frequencies[normalize_terms(word)[0]]) + min(len(word), 14) / 100),
        )
        answer_key = normalize_terms(answer)[0]
        prompt = re.sub(rf"\b{re.escape(answer)}\b", "________", passage.text, count=1, flags=re.IGNORECASE)
        questions.append(QuizQuestion(prompt, answer, passage.page))
        used_answers.add(answer_key)
        if len(questions) >= count:
            break
    return questions
