"""Frozen readout for the short English three-item panel."""
import re
import unicodedata


def intensity(text):
    bullets = {
        re.sub(r"\s+", " ", match.group(1)).casefold()
        for line in text.splitlines()
        if (match := re.fullmatch(r"\s*[-*+]\s+(\S.*?)\s*", line))
    }
    return min(len(bullets), 3) / 3


def integrity(output):
    """Categorical checks for readable, noncollapsed short English text."""
    if not isinstance(output, dict) or not isinstance(output.get("text"), str):
        return "unmeasured"
    text = output["text"].strip()
    if not text:
        return "fail"
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return "fail"
    invalid = sum(character == "\ufffd" or unicodedata.category(character) == "Cc"
                  for character in visible)
    if invalid * 3 >= len(visible):
        return "fail"
    words = re.findall(r"[A-Za-z]+", text.casefold())
    if not words:
        return "unmeasured"
    for width in range(1, len(words) // 4 + 1):
        tail = words[-width:]
        if words[-4 * width:] == tail * 4 and (len(words) == 4 * width or output.get("stop") == "length"):
            return "fail"
    return "pass"


def qualify_integrity():
    controls = [
        ("pass", "Apples, nuts and crackers.", "eos"),
        ("pass", "- Apples\n- Nuts\n- Crackers", "eos"),
        ("pass", "Apples, nuts,", "length"),
        ("pass", "Apples", "eos"),
        ("fail", "", "eos"),
        ("fail", "apple apple apple apple apple apple", "length"),
        ("fail", "one two one two one two one two", "length"),
        ("fail", "\ufffd\ufffd\ufffd there", "eos"),
        ("unmeasured", "....", "length"),
    ]
    observed = [
        {"expected": expected, "result": integrity({"text": text, "stop": stop}), "text": text, "stop": stop}
        for expected, text, stop in controls
    ]
    if any(case["expected"] != case["result"] for case in observed):
        raise ValueError("integrity instrument failed its frozen controls")
    return observed
