import re


# Historical agent initials/signatures seen in GloCare responses.
AGENT_SIGNATURES = {
    "AP",
    "CA",
    "SO",
    "TT",
    "GA",
}


# Deterministic replacements for unsupported operational/capability claims.
#
# These patterns target claims that imply the current system has performed
# an internal action or has access to unavailable GloCare systems.
CAPABILITY_REPLACEMENTS = [
    (
        r"\bthis will help us investigate(?: further)?\b",
        "this will help us better understand the issue",
    ),
    (
        r"\bthis will help us investigate(?: and)? resolve\b",
        "this will help us better understand",
    ),
    (
        r"\bwe will investigate(?: further)?\b",
        "we will review the information provided",
    ),
    (
        r"\bwe(?:'ve| have) investigated\b",
        "we have reviewed the information provided",
    ),
    (
        r"\bwe(?:'ve| have) checked your account\b",
        "please share the relevant account details",
    ),
    (
        r"\bwe(?:'ve| have) verified your account\b",
        "please share the relevant account details",
    ),
    (
        r"\bwe(?:'ve| have) verified your transaction\b",
        "please share the relevant transaction details",
    ),
    (
        r"\bwe(?:'ve| have) confirmed your transaction\b",
        "please share the relevant transaction details",
    ),
    (
        r"\bwe(?:'ve| have) confirmed your recharge\b",
        "please share the relevant recharge details",
    ),
    (
        r"\bwe(?:'ve| have) checked your (?:payment|recharge|subscription)\b",
        "please share the relevant details",
    ),
    (
        r"\bthe issue has been escalated\b",
        "please share the relevant details so the issue can be reviewed",
    ),
    (
        r"\byour complaint has been escalated\b",
        "please share the relevant details so the issue can be reviewed",
    ),
    (
        r"\bour (?:network )?engineers are (?:currently )?(?:working|addressing|investigating) (?:on|the)\b",
        "please share the relevant details so we can better understand",
    ),
    (
        r"\bthe issue is being investigated\b",
        "please share the relevant details so we can better understand the issue",
    ),
    (
        r"\bthe issue is being resolved\b",
        "please share the relevant details so we can better understand the issue",
    ),
    (
        r"\bthe issue has been resolved\b",
        "please let us know if you are still experiencing the issue",
    ),
    (
        r"\byour service has been restored\b",
        "please let us know if you are still experiencing the issue",
    ),
    (
        r"\byour service has been activated\b",
        "please share the relevant details so we can assist",
    ),
    (
        r"\byour service has been deactivated\b",
        "please share the relevant service details",
    ),
    (
        r"\bwe(?:'ve| have) processed your refund\b",
        "please share the relevant refund details",
    ),
    (
        r"\bwe(?:'ve| have) activated your (?:plan|service)\b",
        "please share the relevant plan or service details",
    ),
    (
        r"\bwe(?:'ve| have) deactivated your (?:plan|service)\b",
        "please share the relevant plan or service details",
    ),
    (
        r"\bwe(?:'ve| have) restored your service\b",
        "please let us know if you are still experiencing the issue",
    ),
    (
        r"\bwe(?:'ve| have) resolved (?:the|your) issue\b",
        "please let us know if you are still experiencing the issue",
    ),
    (
        r"\bwe(?:'ll| will) resolve this (?:quickly|shortly|promptly)\b",
        "we will use the information provided to better understand the issue",
    ),
    (
        r"\bwe(?:'ll| will) (?:investigate|check|verify) (?:this|the issue|your account)\b",
        "please share the relevant details so we can better understand the issue",
    ),
    (
        r"\binvestigated the issue\b",
        "reviewed the information provided",
    ),
]


def _apply_capability_firewall(reply):
    """
    Replace unsupported claims about internal actions, account state,
    investigation, escalation, resolution, or service changes.
    """

    for pattern, replacement in CAPABILITY_REPLACEMENTS:
        reply = re.sub(
            pattern,
            replacement,
            reply,
            flags=re.IGNORECASE,
        )

    return reply


def sanitize_reply(reply):
    """
    Deterministically sanitize an LLM-generated customer-facing reply.

    The sanitizer:
    1. removes historical Twitter/agent artifacts,
    2. removes URLs,
    3. removes known historical signatures,
    4. removes unsupported operational/capability claims,
    5. normalizes whitespace.
    """

    # Remove Twitter-style usernames.
    reply = re.sub(r"@\w+", "", reply)

    # Remove URLs, including t.co historical links.
    reply = re.sub(r"https?://\S+", "", reply)

    # Remove known agent signatures when they appear at the end.
    signature_pattern = (
        r"\s+(?:"
        + "|".join(AGENT_SIGNATURES)
        + r")\s*$"
    )

    reply = re.sub(
        signature_pattern,
        "",
        reply,
        flags=re.IGNORECASE,
    )

    # Deterministic capability firewall.
    reply = _apply_capability_firewall(reply)

    # Normalize excessive whitespace.
    reply = re.sub(r"\s+", " ", reply)

    # Remove whitespace before punctuation.
    reply = re.sub(r"\s+([,.!?])", r"\1", reply)

    return reply.strip()