"""NLP utilities for the NLP Study Assistant mini-project."""

from __future__ import annotations

import re
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from functools import lru_cache

from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
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
TERM_SYNONYMS = {
    "change": "convert", "transform": "convert", "turn": "convert",
    "make": "produce", "create": "produce", "build": "create",
    "catch": "capture", "trap": "capture",
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
    source_text: str = ""


@dataclass(frozen=True)
class Flashcard:
    front: str
    back: str
    page: int | None = None


@dataclass(frozen=True)
class MultipleChoiceQuestion:
    prompt: str
    answer: str
    choices: tuple[str, ...]
    page: int | None = None


@dataclass(frozen=True)
class TopicTerm:
    term: str
    score: float
    frequency: int


@dataclass(frozen=True)
class ConceptFact:
    term: str
    text: str
    page: int | None


@dataclass(frozen=True)
class _RetrievalModel:
    vectorizer: TfidfVectorizer
    passage_matrix: object
    svd: TruncatedSVD | None
    semantic_passages: object | None


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
    text = re.sub(r"\s*[•▪◦]\s*", ". ", text)

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


def is_code_like(text: str) -> bool:
    """Keep code fragments out of summaries, retrieval, and generated quizzes."""
    if re.search(r"\b(?:SET|GET)\s+[A-Za-z_]\w*\s*\[", text, re.IGNORECASE):
        return True
    if re.search(r"\bINSERT\s+INTO\b|https?://|\bwww\.", text, re.IGNORECASE):
        return True
    if text.count("=") >= 2 and re.search(r"[\[\]{}]", text):
        return True
    return False


def normalize_terms(text: str) -> list[str]:
    """Drop question framing and apply small, transparent suffix normalization."""
    # Splitting hyphenated terms lets "column-family" match "column family".
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
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
        token = TERM_SYNONYMS.get(token, token)
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


@lru_cache(maxsize=8)
def _fit_retrieval_model(passage_texts: tuple[str, ...]) -> _RetrievalModel | None:
    """Fit a cached TF-IDF + LSA model for one document's readable passages."""
    if not passage_texts:
        return None
    vectorizer = TfidfVectorizer(
        tokenizer=normalize_terms,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(passage_texts)
    except ValueError:
        return None

    svd: TruncatedSVD | None = None
    semantic_passages = None
    component_count = min(100, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if component_count >= 1:
        try:
            svd = TruncatedSVD(n_components=component_count, n_iter=7, random_state=42)
            semantic_passages = svd.fit_transform(matrix)
        except (ValueError, RuntimeError):
            svd = None
            semantic_passages = None
    return _RetrievalModel(vectorizer, matrix, svd, semantic_passages)


def summarize(pages: list[SourcePage], sentence_count: int = 3) -> list[Passage]:
    passages = [item for item in split_passages(pages) if not is_code_like(item.text)]
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
    passages = [item for item in split_passages(pages) if not is_code_like(item.text)]
    if not passages or not question.strip():
        return AnswerResult("", (), False)

    if is_document_overview_question(question):
        overview = tuple(ScoredPassage(passage, 1.0) for passage in summarize(pages, 3))
        return AnswerResult("Overview from your material", overview, bool(overview), overview=True)

    question_terms = set(normalize_terms(question))
    if not question_terms:
        return AnswerResult("", (), False)
    model = _fit_retrieval_model(tuple(passage.text for passage in passages))
    if model is None:
        return AnswerResult("", (), False)
    query_vector = model.vectorizer.transform([question])
    lexical_scores = cosine_similarity(query_vector, model.passage_matrix).ravel()
    semantic_scores = [0.0] * len(passages)
    if model.svd is not None and model.semantic_passages is not None:
        try:
            query_semantics = model.svd.transform(query_vector)
            semantic_scores = cosine_similarity(query_semantics, model.semantic_passages).ravel().tolist()
        except ValueError:
            semantic_scores = [0.0] * len(passages)

    scored: list[ScoredPassage] = []
    overlaps: list[int] = []
    for passage, lexical_score, semantic_score in zip(passages, lexical_scores, semantic_scores):
        sentence_terms = set(normalize_terms(passage.text))
        overlap = len(question_terms & sentence_terms)
        if overlap or semantic_score >= 0.18:
            coverage = overlap / len(question_terms)
            semantic_score = max(0.0, float(semantic_score))
            score = 0.60 * float(lexical_score) + 0.30 * semantic_score + 0.10 * coverage
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


def _extract_concept_facts(pages: list[SourcePage]) -> list[ConceptFact]:
    """Extract simple definition and subject-action facts, skipping code fragments."""
    facts: list[ConceptFact] = []
    seen: set[str] = set()
    generic_subjects = {"it", "this", "that", "they", "we", "you", "plant", "plants", "system", "data"}
    definition_verbs = re.compile(r"\b(is|are|means|refers\s+to|denotes|is\s+defined\s+as|are\s+defined\s+as)\b", re.I)
    action_verbs = re.compile(
        r"\b(allow|allows|enable|enables|absorb|absorbs|store|stores|convert|converts|"
        r"produce|produces|contain|contains|include|includes|represent|represents|"
        r"classify|classifies|detect|detects|measure|measures|use|uses|"
        r"provide|provides|prevent|prevents|support|supports)\b",
        re.I,
    )

    for passage in split_passages(pages):
        text = passage.text.strip(" \t\r\n•·")
        if is_code_like(text) or len(text.split()) < 5:
            continue

        definition_match = definition_verbs.search(text)
        if definition_match:
            subject = text[:definition_match.start()].strip(" .:;–—-")
        else:
            action_matches = list(action_verbs.finditer(text))
            action_match = next(
                (
                    match for match in action_matches
                    if not (
                        match.group(0).casefold().startswith("store")
                        and re.match(r"\s*,\s*(?:such\s+as|including)\b", text[match.end():], re.I)
                    )
                ),
                None,
            )
            if not action_match:
                continue
            subject = text[:action_match.start()].strip(" .:;–—-")
            # For lists of examples, retain the named category before the examples.
            subject = re.split(r",\s*(?:such\s+as|including|for\s+example)?\b", subject, maxsplit=1, flags=re.I)[0]

        subject = re.sub(r"^(?:chapter|section)\s+[\w.-]+\s*[:.]?\s*", "", subject, flags=re.I)
        subject = re.sub(r"^(?:a|an|the)\s+", "", subject, flags=re.I).strip()
        subject_words = subject.split()
        if not subject_words or len(subject_words) > 6:
            continue
        normalized = normalize_terms(subject)
        if not normalized or len(normalized) > 6 or all(word in generic_subjects for word in normalized):
            continue
        key = " ".join(normalized)
        if key in seen:
            continue
        facts.append(ConceptFact(subject, text, passage.page))
        seen.add(key)
    return facts


def generate_quiz(pages: list[SourcePage], count: int = 5) -> list[QuizQuestion]:
    """Create concept recall questions from definition-like source statements."""
    if count <= 0:
        return []
    facts = _extract_concept_facts(pages)
    return [
        QuizQuestion(
            f"Which concept is described by this statement?\n\n{fact.text}",
            fact.term,
            fact.page,
            fact.text,
        )
        for fact in facts[:count]
    ]


def generate_flashcards(pages: list[SourcePage], count: int = 10) -> list[Flashcard]:
    """Make concept-to-definition flashcards from the source."""
    if count <= 0:
        return []
    quiz_items = generate_quiz(pages, count)
    return [
        Flashcard(item.answer, item.source_text or item.prompt.replace("________", item.answer), item.page)
        for item in quiz_items
    ]


def generate_multiple_choice_quiz(
    pages: list[SourcePage], count: int = 5, choice_count: int = 4
) -> list[MultipleChoiceQuestion]:
    """Turn concept-definition questions into deterministic multiple-choice questions."""
    if count <= 0 or choice_count < 2:
        return []
    questions = generate_quiz(pages, count)
    if len(questions) < 2:
        return []

    terms = [question.answer for question in questions]
    unique_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = _answer_key(term)
        if key and key not in seen:
            unique_terms.append(term)
            seen.add(key)

    result: list[MultipleChoiceQuestion] = []
    for item in questions:
        distractors = [term for term in unique_terms if _answer_key(term) != _answer_key(item.answer)]
        rng = random.Random(f"{item.prompt}|{item.answer}")
        rng.shuffle(distractors)
        choices = [item.answer, *distractors[:min(choice_count, len(unique_terms)) - 1]]
        rng.shuffle(choices)
        result.append(MultipleChoiceQuestion(item.prompt, item.answer, tuple(choices), item.page))
    return result


def extract_key_terms(pages: list[SourcePage], limit: int = 12) -> list[TopicTerm]:
    """Rank useful words and short phrases with corpus TF-IDF and frequency."""
    if limit <= 0:
        return []
    passages = [item for item in split_passages(pages) if not is_code_like(item.text)]
    if not passages:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform([passage.text for passage in passages])
    except ValueError:
        return []

    terms = vectorizer.get_feature_names_out()
    scores = matrix.sum(axis=0).A1
    raw_counts: Counter[str] = Counter()
    for passage in passages:
        raw_counts.update(re.findall(r"(?u)\b[a-zA-Z]{3,}\b", passage.text.casefold()))

    ranked: list[TopicTerm] = []
    for term, score in zip(terms, scores):
        words = term.casefold().split()
        if any(word in QUIZ_EXCLUDED_WORDS or word in QUESTION_FILLER_WORDS for word in words):
            continue
        if len(words) == 1 and (words[0] in ENGLISH_STOP_WORDS or len(words[0]) < 4):
            continue
        frequency = min(raw_counts[word] for word in words)
        ranked.append(TopicTerm(term, float(score), frequency))
    ranked.sort(key=lambda item: (item.score, item.frequency, len(item.term)), reverse=True)

    # Keep the highest scoring phrase or word for each normalized lead term.
    selected: list[TopicTerm] = []
    covered: set[str] = set()
    for item in ranked:
        normalized = normalize_terms(item.term)
        if not normalized or normalized[0] in covered:
            continue
        selected.append(item)
        covered.add(normalized[0])
        if len(selected) >= limit:
            break
    return selected
