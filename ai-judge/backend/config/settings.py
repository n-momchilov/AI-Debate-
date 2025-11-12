"""Global configuration constants for AI Judge.

These values centralize model, token/word limits, and file paths.
"""

# Model settings
MODEL_NAME = "llama3:8b"
TEMPERATURES = {
    "emotional": 0.8,
    "logical": 0.25,
    "judge": 0.0,
}

# Word limits
WORD_LIMIT_MIN = 250
WORD_LIMIT_MAX = 350
JUDGE_WORD_LIMIT_MIN = 300
JUDGE_WORD_LIMIT_MAX = 400

# Token limits (word count * 1.33 for approximation)
MAX_TOKENS_ARGUMENT = 470  # 350 words * 1.33
# Bump to allow more headroom for strict JSON + reasoning
MAX_TOKENS_VERDICT = 700   # previously 530; reduces truncation risk

# Debate settings
NUM_ROUNDS = 3

# File paths
DEBATES_FILE = "backend/data/debates.json"
STATISTICS_FILE = "backend/data/statistics.json"
