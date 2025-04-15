from jiwer import wer
from typing import List

def compute_wer(reference: List[str], hypothesis: List[str]) -> float:
    """
    Computes average Word Error Rate between two lists of sentences.
    
    Args:
        reference (List[str]): List of ground truth sentences.
        hypothesis (List[str]): List of predicted/generated sentences.
    
    Returns:
        float: Average WER across all pairs.
    """
    if len(reference) != len(hypothesis):
        raise ValueError("Reference and hypothesis must be the same length")

    scores = [wer(r, h) for r, h in zip(reference, hypothesis)]
    return sum(scores) / len(scores)

def load_transcripts(ref_path: str, hyp_path: str) -> tuple:
    """
    Loads text files containing reference and hypothesis transcripts.
    
    Args:
        ref_path (str): Path to file with reference (ground truth) sentences.
        hyp_path (str): Path to file with hypothesis (predicted) sentences.

    Returns:
        tuple: Two lists of strings (reference, hypothesis)
    """
    with open(ref_path, "r", encoding="utf-8") as f:
        reference = [line.strip() for line in f]

    with open(hyp_path, "r", encoding="utf-8") as f:
        hypothesis = [line.strip() for line in f]

    return reference, hypothesis
