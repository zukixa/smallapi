import random
from heapq import nlargest
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

async def progressive_summarize_text(
    text, max_length, initial_reduction_ratio=0.8, step=0.1
):
    if len(text) < max_length:
        return text
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Remove stopwords and apply stemming
    words = [ps.stem(word) for word in words if word not in stop_words]

    # Calculate word frequencies
    word_freqs = FreqDist(words)

    # Score sentences based on word frequency
    sentence_scores = {}
    for sentence_idx, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_freqs.keys():
                sentence_scores[sentence_idx] = (
                    sentence_scores.get(sentence_idx, 0) + word_freqs[word]
                )

    reduction_ratio = initial_reduction_ratio

    # Extract top N sentences based on their scores
    while True:
        num_sentences_to_summarize = round(len(sentences) * reduction_ratio)
        if num_sentences_to_summarize < 1:
            num_sentences_to_summarize = 1

        selected_sentence_indexes = nlargest(
            num_sentences_to_summarize, sentence_scores, key=sentence_scores.get
        )

        # Combine selected sentences to make the summary
        summary = " ".join(
            [sentences[idx] for idx in sorted(selected_sentence_indexes)]
        )

        if 0 < len(summary.strip()) <= max_length:
            break
        elif reduction_ratio - step < 0:
            selected_sentence_indexes = nlargest(
                1, sentence_scores, key=sentence_scores.get
            )

            # Combine the single highest-scoring sentence as the last resort
            summary = " ".join(
                [sentences[idx] for idx in sorted(selected_sentence_indexes)]
            )
            break
        else:
            reduction_ratio -= step

    return summary

def combine_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    combined_message = "\n\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages
    )
    return [{"role": "user", "content": combined_message}]


def validate_messages_format(messages):
    if not messages:
        return False
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        return False
    if messages[0]["role"] == "assistant" or messages[-1]["role"] == "assistant":
        return False
    if any(message["role"] == "system" for message in messages[1:]):
        return False
    return True

def generate_random_string(length):
    return "".join(
        random.choices(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=length
        )
    )

def chunk_string(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def stringify_messages(messages):
    output = ""
    for message in messages[:-1]:
        role = message["role"]
        content = message["content"]

        prefix = ""
        if role == "user":
            prefix = "USR: "
        elif role == "assistant":
            prefix = "ASSISTNT: "
        elif role == "system":
            prefix = "SYSTM: "

        output += prefix + content + "\n"

    return output.strip() + "\nUSR: " + messages[-1]["content"]

