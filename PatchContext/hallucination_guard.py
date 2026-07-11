import re
import string
from typing import List, Tuple, Set

# Standard English stopwords
STOPWORDS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now"
}

def clean_text(text: str) -> List[str]:
    """Normalize text: convert to lowercase, remove punctuation, and remove stopwords."""
    if not text:
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace punctuation with spaces
    # We replace punctuation with spaces to avoid joining words incorrectly
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Split on whitespace and filter out empty strings and stopwords
    words = [w for w in text.split() if w and w not in STOPWORDS]
    
    return words

def check_grounding(answer: str, source_contents: List[str], threshold: float = 0.15) -> Tuple[bool, float, str]:
    """
    Check if the generated answer is grounded in the retrieved source documents.
    
    Returns:
        is_grounded (bool): True if score >= threshold or fallback match
        score (float): Grounding score between 0.0 and 1.0
        verdict (str): 'Safe', 'Warning', or 'Hallucinated'
    """
    fallback_string = "I couldn't find this in repository history."
    
    # Direct check or normalized check for the fallback response
    answer_clean_str = " ".join(clean_text(answer))
    fallback_clean_str = " ".join(clean_text(fallback_string))
    
    if answer.strip().strip('"').strip("'") == fallback_string or answer_clean_str == fallback_clean_str:
        return True, 1.0, "Safe"
        
    answer_words = clean_text(answer)
    if not answer_words:
        # If there are no words left after cleaning, check raw words
        raw_words = [w.lower() for w in answer.split() if w]
        if not raw_words:
            return False, 0.0, "Hallucinated"
        answer_words = raw_words

    # Build set of words from source documents
    source_words_set = set()
    for doc in source_contents:
        source_words_set.update(clean_text(doc))
        
    if not source_words_set:
        return False, 0.0, "Hallucinated"
        
    # Calculate overlap
    overlapping_words = [w for w in answer_words if w in source_words_set]
    score = len(overlapping_words) / len(answer_words)
    
    is_grounded = score >= threshold
    
    if score >= threshold:
        verdict = "Safe"
    elif score >= 0.05:
        verdict = "Warning"
    else:
        verdict = "Hallucinated"
        
    return is_grounded, score, verdict

if __name__ == "__main__":
    # Local tests
    test_ans1 = "I couldn't find this in repository history."
    print(check_grounding(test_ans1, []))
    
    test_ans2 = "FastAPI uses Pydantic for data validation."
    test_sources = ["FastAPI is a fast web framework. It uses Pydantic for request and response validation."]
    print(check_grounding(test_ans2, test_sources))
    
    test_ans3 = "Flask is a micro web framework written in Python."
    print(check_grounding(test_ans3, test_sources))
