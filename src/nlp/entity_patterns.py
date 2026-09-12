"""Regex patterns for URL masking and entity preservation in NLP normalization (M1.P1.2.F2).

Defines compiled patterns for detecting URLs, preserving device models,
recognizing operating system versions, and protecting pre-masked PII tokens.
"""

import re
from typing import List

# ---------------------------------------------------------------------------
# 1. URL Patterns
# ---------------------------------------------------------------------------
# Matches:
# - Standard http:// and https:// URLs
# - Standard www. URLs
# - Common bare domain URLs with paths (e.g., support.apple.com/iphone-x, apple.co/help)
# - Twitter t.co shortener links (including truncated ones mid-token, e.g. t.co/xyz123)
# Excludes:
# - Trailing sentence punctuation (.,!?:;)[]"')
# - Placeholder Unicode private-use codepoints (\ue000-\ue00f)
_URL_PATTERN_STR = (
    r"(?:"
    r"(?:https?://|www\.)[^\s\ue000-\ue00f]*[^\s\ue000-\ue00f.,!?:;)\]\"']"
    r"|"
    r"(?:\b|\A)(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|co|io|app|apple|ly)/[^\s\ue000-\ue00f]*[^\s\ue000-\ue00f.,!?:;)\]\"']"
    r"|"
    r"(?:\b|\A)t\.co/[a-zA-Z0-9_]+"
    r")"
)

URL_PATTERN = re.compile(_URL_PATTERN_STR, flags=re.IGNORECASE)

# ---------------------------------------------------------------------------
# 2. Device Model & Operating System Patterns
# ---------------------------------------------------------------------------
# Preserves device models and OS version strings critical for BM25 exact-token
# and dense retrieval. Must be case-insensitive, space-tolerant, and avoid
# false-positive collisions with common English vocabulary.

DEVICE_PATTERNS: List[re.Pattern[str]] = [
    # iPhone: e.g. iPhone, iPhone X, iPhone 8, iPhone 8 Plus, iPhone 11 Pro Max, iPhone 12 mini, iPhone SE, iPhone 6s, iPhone 7+
    re.compile(
        r"\biPhone(?:\s*(?:[0-9]+[A-Za-z]*|[A-Z]+))?(?:\s*(?:Pro\s*Max|Pro|Plus|Max|Mini|s|S|\+))?(?!\w)",
        flags=re.IGNORECASE,
    ),
    # iPad: e.g. iPad, iPad Pro, iPad Air, iPad Mini, iPad 2, iPad Air 2, iPad mini 4, iPad Pro 10.5
    re.compile(
        r"\biPad(?:\s*(?:Pro|Air|Mini))?(?:\s*[0-9]+(?:\.[0-9]+)?)?\b",
        flags=re.IGNORECASE,
    ),
    # Apple Watch: e.g. Apple Watch, Apple Watch Series 3, Apple Watch Ultra, Apple Watch SE
    re.compile(
        r"\bApple\s*Watch(?:\s*(?:Series\s*[0-9]+|Ultra(?:\s*[0-9]+)?|SE))?\b",
        flags=re.IGNORECASE,
    ),
    # Mac computers: e.g. MacBook, MacBook Pro, MacBook Air, iMac, iMac Pro, Mac mini, Mac Pro, Mac Studio
    re.compile(
        r"\b(?:MacBook(?:\s*(?:Pro|Air))?|iMac(?:\s*Pro)?|Mac\s*(?:mini|Pro|Studio))\b",
        flags=re.IGNORECASE,
    ),
    # Named macOS releases: e.g. macOS High Sierra, macOS Sierra, macOS Big Sur, macOS Monterey
    re.compile(
        r"\bmacOS\s+(?:High\s+Sierra|Sierra|El\s+Capitan|Big\s+Sur|Sonoma|Ventura|Monterey|Catalina|Mojave|Sequoia|Yosemite|Mavericks)(?:\s+[0-9]+(?:\.[0-9]+)*)?\b",
        flags=re.IGNORECASE,
    ),
    # Operating system versions with numbers: e.g. iOS 11.1, ios 11.1, IOS 11.1, iOS11.1, macOS 10.13, watchOS 4, tvOS 11, iPadOS 16
    re.compile(
        r"\b(?:iOS|macOS|watchOS|tvOS|iPadOS)(?:\s*[0-9]+(?:\.[0-9]+)*)?\b",
        flags=re.IGNORECASE,
    ),
    # Hardware / Model codes: e.g. SM-T280, SM-G950F (requires hyphen to prevent matching words like smart, small, smoke)
    re.compile(
        r"\bSM-[A-Za-z0-9]+\b",
        flags=re.IGNORECASE,
    ),
]

# Aliases for explicit and extensible pattern lists
DEVICE_MODEL_PATTERNS = DEVICE_PATTERNS

# Combined pattern for convenience and compatibility
DEVICE_MODEL_PATTERN = re.compile(
    "|".join(f"(?:{p.pattern})" for p in DEVICE_PATTERNS),
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 3. Pre-masked PII Patterns
# ---------------------------------------------------------------------------
# Dataset pre-masked placeholders like __email__ and __phone__ that must
# survive normalization completely untouched, even when adjacent to punctuation.
PII_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"__(?:email|phone)__", flags=re.IGNORECASE),
]

# Combined list of all protected patterns in order of precedence
ALL_PROTECTED_PATTERNS: List[re.Pattern[str]] = [*DEVICE_PATTERNS, *PII_PATTERNS]
