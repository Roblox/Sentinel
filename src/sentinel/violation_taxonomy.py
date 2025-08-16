# Copyright 2025 Roblox Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive violation taxonomy for multi-platform safety detection.

This module defines a structured taxonomy of content violations across different platforms,
providing standardized categories for training data organization and detection model configuration.
"""

from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass


class ViolationType(Enum):
    """Core violation types that apply across platforms."""
    
    # Safety & Harm
    CHILD_SAFETY = "child_safety"
    VIOLENCE_THREATS = "violence_threats"
    SELF_HARM = "self_harm"
    DANGEROUS_ACTIVITIES = "dangerous_activities"
    
    # Harassment & Bullying
    HARASSMENT = "harassment"
    CYBERBULLYING = "cyberbullying"
    DOXXING = "doxxing"
    TARGETED_ABUSE = "targeted_abuse"
    
    # Hate Speech & Discrimination
    HATE_SPEECH = "hate_speech"
    DISCRIMINATION = "discrimination"
    EXTREMISM = "extremism"
    RADICALIZATION = "radicalization"
    
    # Sexual Content
    SEXUAL_CONTENT = "sexual_content"
    SEXUAL_EXPLOITATION = "sexual_exploitation"
    GROOMING = "grooming"
    
    # Platform Manipulation
    SPAM = "spam"
    SCAMS = "scams"
    PHISHING = "phishing"
    IMPERSONATION = "impersonation"
    COORDINATED_ABUSE = "coordinated_abuse"
    
    # Gaming-Specific
    GRIEFING = "griefing"
    CHEATING_COORDINATION = "cheating_coordination"
    GAME_TOXICITY = "game_toxicity"
    
    # Platform Circumvention
    FILTER_BYPASS = "filter_bypass"
    BAN_EVASION = "ban_evasion"
    RULE_CIRCUMVENTION = "rule_circumvention"
    
    # General/Fallback
    GENERAL_HATE = "general_hate"


class Platform(Enum):
    """Supported platforms with specific characteristics."""
    
    DISCORD = "discord"
    ROBLOX = "roblox"
    TWITTER = "twitter"
    REDDIT = "reddit"
    TWITCH = "twitch"
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    GENERAL = "general"


@dataclass
class ViolationCategory:
    """Detailed violation category with metadata."""
    
    violation_type: ViolationType
    name: str
    description: str
    severity: int  # 1-5 scale, 5 being most severe
    platforms: Set[Platform]
    examples: List[str]
    keywords: List[str]
    detection_patterns: List[str]


# Comprehensive violation taxonomy
VIOLATION_TAXONOMY: Dict[ViolationType, ViolationCategory] = {
    
    # Child Safety
    ViolationType.CHILD_SAFETY: ViolationCategory(
        violation_type=ViolationType.CHILD_SAFETY,
        name="Child Safety Violations",
        description="Content that endangers, exploits, or inappropriately targets minors",
        severity=5,
        platforms={Platform.ROBLOX, Platform.DISCORD, Platform.YOUTUBE, Platform.TIKTOK, Platform.GENERAL},
        examples=[
            "asking for personal information from minors",
            "inappropriate roleplay with children",
            "sharing content unsuitable for minors",
            "attempting to establish private contact with children"
        ],
        keywords=["age", "young", "kid", "child", "school", "parent", "address", "phone"],
        detection_patterns=[
            "age verification attempts",
            "requests for personal information",
            "inappropriate adult-child interactions"
        ]
    ),
    
    # Grooming
    ViolationType.GROOMING: ViolationCategory(
        violation_type=ViolationType.GROOMING,
        name="Online Grooming",
        description="Predatory behavior designed to build trust with minors for exploitation",
        severity=5,
        platforms={Platform.ROBLOX, Platform.DISCORD, Platform.INSTAGRAM, Platform.TIKTOK, Platform.GENERAL},
        examples=[
            "requesting to move conversation to private platform",
            "offering gifts or rewards to children",
            "gradually introducing sexual topics to minors",
            "asking children to keep conversations secret"
        ],
        keywords=["secret", "private", "special", "gift", "reward", "meet", "alone"],
        detection_patterns=[
            "escalating personal questions",
            "offers of benefits or rewards",
            "secrecy requests",
            "platform migration attempts"
        ]
    ),
    
    # Harassment
    ViolationType.HARASSMENT: ViolationCategory(
        violation_type=ViolationType.HARASSMENT,
        name="Harassment and Abuse",
        description="Repeated unwanted contact or abusive behavior toward individuals",
        severity=4,
        platforms={Platform.DISCORD, Platform.TWITTER, Platform.REDDIT, Platform.TWITCH, Platform.GENERAL},
        examples=[
            "repeated unwanted messages or mentions",
            "coordinated harassment campaigns",
            "stalking behavior across platforms",
            "malicious tagging or spam mentions"
        ],
        keywords=["kill yourself", "die", "worthless", "pathetic", "loser", "stupid"],
        detection_patterns=[
            "repeated negative interactions",
            "coordinated group behavior",
            "escalating aggressive language"
        ]
    ),
    
    # Hate Speech
    ViolationType.HATE_SPEECH: ViolationCategory(
        violation_type=ViolationType.HATE_SPEECH,
        name="Hate Speech",
        description="Content attacking individuals or groups based on protected characteristics",
        severity=4,
        platforms={Platform.TWITTER, Platform.REDDIT, Platform.DISCORD, Platform.YOUTUBE, Platform.GENERAL},
        examples=[
            "racial slurs and epithets",
            "religious hatred and intolerance",
            "lgbtq+ discrimination and slurs",
            "xenophobic and nationalist rhetoric"
        ],
        keywords=["slur", "inferior", "subhuman", "genocide", "ethnic cleansing"],
        detection_patterns=[
            "dehumanizing language",
            "supremacist ideology",
            "calls for violence against groups"
        ]
    ),
    
    # Gaming Toxicity
    ViolationType.GAME_TOXICITY: ViolationCategory(
        violation_type=ViolationType.GAME_TOXICITY,
        name="Gaming Toxicity",
        description="Toxic behavior specific to gaming environments",
        severity=3,
        platforms={Platform.DISCORD, Platform.TWITCH, Platform.ROBLOX, Platform.GENERAL},
        examples=[
            "intentional feeding or throwing games",
            "rage quitting and toxic rage",
            "stream sniping coordination",
            "competitive integrity violations"
        ],
        keywords=["noob", "trash", "feeding", "throwing", "rage quit", "stream snipe"],
        detection_patterns=[
            "gaming-specific insults",
            "competitive manipulation",
            "intentional game disruption"
        ]
    ),
    
    # Filter Bypass
    ViolationType.FILTER_BYPASS: ViolationCategory(
        violation_type=ViolationType.FILTER_BYPASS,
        name="Filter Bypass",
        description="Attempts to circumvent platform content filters and moderation",
        severity=3,
        platforms={Platform.ROBLOX, Platform.DISCORD, Platform.REDDIT, Platform.GENERAL},
        examples=[
            "character substitution in prohibited words",
            "zalgo text and unicode manipulation",
            "spacing letters to avoid detection",
            "code words and dog whistles"
        ],
        keywords=["bypass", "filter", "censor", "moderate"],
        detection_patterns=[
            "character substitution patterns",
            "unusual spacing or formatting",
            "unicode manipulation",
            "coded language"
        ]
    ),
    
    # Scams and Fraud
    ViolationType.SCAMS: ViolationCategory(
        violation_type=ViolationType.SCAMS,
        name="Scams and Fraud",
        description="Fraudulent schemes designed to steal money, items, or personal information",
        severity=4,
        platforms={Platform.DISCORD, Platform.TWITTER, Platform.INSTAGRAM, Platform.GENERAL},
        examples=[
            "fake giveaways and contests",
            "cryptocurrency and nft scams",
            "phishing for account credentials",
            "fake trading and marketplace scams"
        ],
        keywords=["free", "giveaway", "winner", "click link", "verify", "urgent"],
        detection_patterns=[
            "urgency and time pressure",
            "too-good-to-be-true offers",
            "credential harvesting attempts"
        ]
    ),
    
    # Doxxing
    ViolationType.DOXXING: ViolationCategory(
        violation_type=ViolationType.DOXXING,
        name="Doxxing and Privacy Violations",
        description="Sharing private personal information without consent",
        severity=5,
        platforms={Platform.TWITTER, Platform.REDDIT, Platform.DISCORD, Platform.GENERAL},
        examples=[
            "sharing real names and addresses",
            "posting phone numbers and emails",
            "workplace and school information",
            "family member information"
        ],
        keywords=["address", "phone", "email", "workplace", "school", "family"],
        detection_patterns=[
            "personal information patterns",
            "contact details sharing",
            "location information"
        ]
    ),
    
    # Coordinated Abuse
    ViolationType.COORDINATED_ABUSE: ViolationCategory(
        violation_type=ViolationType.COORDINATED_ABUSE,
        name="Coordinated Abuse",
        description="Organized campaigns of harassment, brigading, or platform manipulation",
        severity=4,
        platforms={Platform.REDDIT, Platform.TWITTER, Platform.DISCORD, Platform.GENERAL},
        examples=[
            "brigading other communities",
            "coordinated reporting abuse",
            "organized harassment campaigns",
            "vote manipulation schemes"
        ],
        keywords=["raid", "brigade", "coordinate", "organize", "mass report"],
        detection_patterns=[
            "coordination language",
            "call to action patterns",
            "organized behavior indicators"
        ]
    ),
    
    # Violence and Threats
    ViolationType.VIOLENCE_THREATS: ViolationCategory(
        violation_type=ViolationType.VIOLENCE_THREATS,
        name="Violence and Threats",
        description="Content containing threats of violence or promoting violent acts",
        severity=5,
        platforms={Platform.TWITTER, Platform.REDDIT, Platform.DISCORD, Platform.GENERAL},
        examples=[
            "direct threats of physical harm",
            "bomb threats and terrorism",
            "school shooting references",
            "assassination and murder threats"
        ],
        keywords=["kill", "murder", "bomb", "shoot", "attack", "violence"],
        detection_patterns=[
            "threat language",
            "violent imagery",
            "specific harm intentions"
        ]
    ),
    
    # General Hate (Fallback)
    ViolationType.GENERAL_HATE: ViolationCategory(
        violation_type=ViolationType.GENERAL_HATE,
        name="General Harmful Content",
        description="General harmful or toxic content that doesn't fit specific categories",
        severity=3,
        platforms={Platform.GENERAL},
        examples=[
            "general toxic behavior",
            "uncategorized harmful content",
            "mixed violation types",
            "platform-agnostic toxicity"
        ],
        keywords=["toxic", "harmful", "inappropriate", "violation"],
        detection_patterns=[
            "general toxicity patterns",
            "mixed harmful indicators",
            "uncategorized violations"
        ]
    )
}


# Platform-specific violation mappings
PLATFORM_VIOLATIONS: Dict[Platform, Set[ViolationType]] = {
    Platform.DISCORD: {
        ViolationType.HARASSMENT, ViolationType.HATE_SPEECH, ViolationType.DOXXING,
        ViolationType.CHILD_SAFETY, ViolationType.GROOMING, ViolationType.SCAMS,
        ViolationType.COORDINATED_ABUSE, ViolationType.GAME_TOXICITY, ViolationType.FILTER_BYPASS
    },
    Platform.ROBLOX: {
        ViolationType.CHILD_SAFETY, ViolationType.GROOMING, ViolationType.HARASSMENT,
        ViolationType.CYBERBULLYING, ViolationType.FILTER_BYPASS, ViolationType.GAME_TOXICITY,
        ViolationType.SEXUAL_CONTENT, ViolationType.SCAMS
    },
    Platform.TWITTER: {
        ViolationType.HATE_SPEECH, ViolationType.HARASSMENT, ViolationType.DOXXING,
        ViolationType.COORDINATED_ABUSE, ViolationType.SCAMS, ViolationType.VIOLENCE_THREATS,
        ViolationType.IMPERSONATION, ViolationType.SPAM
    },
    Platform.REDDIT: {
        ViolationType.HARASSMENT, ViolationType.HATE_SPEECH, ViolationType.COORDINATED_ABUSE,
        ViolationType.DOXXING, ViolationType.SPAM, ViolationType.BAN_EVASION,
        ViolationType.VOTE_MANIPULATION, ViolationType.BRIGADING
    },
    Platform.GENERAL: set(ViolationType)  # All violations apply to general detection
}


def get_violations_for_platform(platform: Platform) -> Set[ViolationType]:
    """Get all violation types relevant to a specific platform."""
    return PLATFORM_VIOLATIONS.get(platform, set())


def get_violation_category(violation_type: ViolationType) -> ViolationCategory:
    """Get detailed information about a specific violation type."""
    return VIOLATION_TAXONOMY.get(violation_type)


def get_high_severity_violations() -> List[ViolationType]:
    """Get violation types with severity level 4 or 5."""
    return [
        vtype for vtype, category in VIOLATION_TAXONOMY.items()
        if category.severity >= 4
    ]


def get_violations_by_keyword(keyword: str) -> List[ViolationType]:
    """Find violation types that include a specific keyword."""
    matches = []
    keyword_lower = keyword.lower()
    
    for vtype, category in VIOLATION_TAXONOMY.items():
        if keyword_lower in [k.lower() for k in category.keywords]:
            matches.append(vtype)
    
    return matches