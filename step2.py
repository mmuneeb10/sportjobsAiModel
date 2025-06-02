import json
import re
import string
import sys
import traceback
import nltk
from nltk.corpus import words as nltk_words
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.metrics.distance import edit_distance
from collections import Counter

# We'll import language_tool_python when needed to avoid errors if it's not installed

# Download required NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/words")
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download("punkt")
    nltk.download("words")
    print("Download complete")


def analyze_cv_quality(json_path):
    """
    Step 2: Quality Analysis - Analyze the CV format and content quality

    Args:
        json_path (str): Path to the JSON file created by step1.py

    Returns:
        dict: Quality analysis scores and feedback
    """
    print("Starting CV quality analysis...")

    # Load the CV data from JSON
    print(f"Loading JSON data from {json_path}...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            cv_data = json.load(f)
        print(f"Successfully loaded JSON with {len(cv_data.get('pages', []))} pages")
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        return {"error": "Failed to load JSON data"}

    # Initialize scores dictionary
    scores = {
        "spelling": {"score": 0, "issues": []},
        "grammar": {"score": 0, "issues": []},
        "formatting": {"score": 0, "issues": []},
        "fonts": {"score": 0, "issues": []},
        "spacing": {"score": 0, "issues": []},
        "content_quality": {"score": 0, "issues": []},
        "overall": 0,
        "feedback": [],
    }
    print("Initialized scores dictionary")

    # Extract all text content for text-based analysis
    print("Extracting text content for analysis...")
    all_text = ""
    all_lines = []

    for page in cv_data.get("pages", []):
        for line_data in page.get("content", []):
            if line_data.get("text"):
                all_text += line_data.get("text") + " "
                all_lines.append(line_data.get("text"))

    print(
        f"Extracted {len(all_lines)} lines of text, total length: {len(all_text)} characters"
    )

    # 1. Spelling check
    print("\nStarting spelling check...")
    try:
        spelling_score, spelling_issues = check_spelling(all_text)
        scores["spelling"]["score"] = spelling_score
        scores["spelling"]["issues"] = spelling_issues
        print(
            f"Spelling check complete. Score: {spelling_score}/10, {len(spelling_issues)} issues found"
        )
    except Exception as e:
        print(f"Error in spelling check: {str(e)}")
        import traceback

        print(traceback.format_exc())
        scores["spelling"]["score"] = 5.0  # Default score
        scores["spelling"]["issues"] = []

    # 2. Grammar check
    print("\nStarting grammar check...")
    try:
        grammar_score, grammar_issues = check_grammar(all_text)
        scores["grammar"]["score"] = grammar_score
        scores["grammar"]["issues"] = grammar_issues
        print(
            f"Grammar check complete. Score: {grammar_score}/10, {len(grammar_issues)} issues found"
        )
    except Exception as e:
        print(f"Error in grammar check: {str(e)}")
        import traceback

        print(traceback.format_exc())
        scores["grammar"]["score"] = 5.0  # Default score
        scores["grammar"]["issues"] = []

    # 3. Formatting consistency check
    print("\nStarting formatting consistency check...")
    try:
        formatting_score, formatting_issues = check_formatting_consistency(cv_data)
        scores["formatting"]["score"] = formatting_score
        scores["formatting"]["issues"] = formatting_issues
        print(
            f"Formatting check complete. Score: {formatting_score}/10, {len(formatting_issues)} issues found"
        )
    except Exception as e:
        print(f"Error in formatting check: {str(e)}")
        import traceback

        print(traceback.format_exc())
        scores["formatting"]["score"] = 5.0  # Default score
        scores["formatting"]["issues"] = []

    # 4. Font evaluation
    print("\nStarting font evaluation...")
    try:
        font_score, font_issues = evaluate_fonts(cv_data)
        scores["fonts"]["score"] = font_score
        scores["fonts"]["issues"] = font_issues
        print(
            f"Font evaluation complete. Score: {font_score}/10, {len(font_issues)} issues found"
        )
    except Exception as e:
        print(f"Error in font evaluation: {str(e)}")
        import traceback

        print(traceback.format_exc())
        scores["fonts"]["score"] = 5.0  # Default score
        scores["fonts"]["issues"] = []

    # 5. Spacing analysis
    print("\nStarting spacing analysis...")
    try:
        spacing_score, spacing_issues = analyze_spacing(cv_data)
        scores["spacing"]["score"] = spacing_score
        scores["spacing"]["issues"] = spacing_issues
        print(
            f"Spacing analysis complete. Score: {spacing_score}/10, {len(spacing_issues)} issues found"
        )
    except Exception as e:
        print(f"Error in spacing analysis: {str(e)}")
        import traceback

        print(traceback.format_exc())
        scores["spacing"]["score"] = 5.0  # Default score
        scores["spacing"]["issues"] = []

    # 6. Content quality analysis
    print("\nStarting content quality analysis...")
    try:
        content_score, content_issues = analyze_content_quality(all_text, all_lines)
        scores["content_quality"]["score"] = content_score
        scores["content_quality"]["issues"] = content_issues
        print(
            f"Content quality analysis complete. Score: {content_score}/10, {len(content_issues)} issues found"
        )
    except Exception as e:
        print(f"Error in content quality analysis: {str(e)}")
        import traceback

        print(traceback.format_exc())
        scores["content_quality"]["score"] = 5.0  # Default score
        scores["content_quality"]["issues"] = []

    # Calculate overall score (weighted average)
    print("\nCalculating overall score...")
    weights = {
        "spelling": 0.15,
        "grammar": 0.15,
        "formatting": 0.20,
        "fonts": 0.15,
        "spacing": 0.15,
        "content_quality": 0.20,
    }

    weighted_sum = sum(
        scores[category]["score"] * weight for category, weight in weights.items()
    )
    scores["overall"] = round(weighted_sum, 1)
    print(f"Overall score calculated: {scores['overall']}/10")

    # Generate overall feedback
    print("\nGenerating feedback...")
    try:
        generate_overall_feedback(scores)
        print(f"Generated {len(scores['feedback'])} feedback items")
    except Exception as e:
        print(f"Error generating feedback: {str(e)}")
        import traceback

        print(traceback.format_exc())
        scores["feedback"] = [
            "CV analysis complete. Some scores may be approximate due to processing limitations."
        ]

    print("\nCV quality analysis complete!")
    return scores


def check_spelling(text):
    """Check for spelling errors in the CV text"""
    print("  Loading English dictionary...")
    try:
        # Initialize English dictionary
        english_words = set(w.lower() for w in nltk_words.words())
        print(f"  Loaded {len(english_words)} English words")
    except Exception as e:
        print(f"  Error loading dictionary: {str(e)}")
        # Fallback to a small set of common words if dictionary fails
        english_words = set(
            ["the", "and", "of", "to", "a", "in", "for", "is", "on", "that", "by"]
        )
        print("  Using minimal fallback dictionary")

    print("  Tokenizing text...")
    # Simpler tokenization fallback to avoid punkt_tab dependency
    tokens = re.findall(r"\b\w+\b", text.lower())

    # Filter out punctuation and numbers
    words = [word for word in tokens if word.isalpha()]
    print(f"  Found {len(words)} words to check")

    # Find misspelled words
    misspelled = []
    count = 0
    for word in words:
        print("THE WORD IS  ", word, "THE COUNT IS", count)
        count += 1
        if len(word) > 2 and word not in english_words:
            # Check if it might be a name or technical term (based on capitalization in original text)
            if not word[0].isupper() and word not in english_words:
                # Find closest word as suggestion
                suggestions = find_closest_words(word, english_words)
                misspelled.append(
                    {
                        "word": word,
                        "suggestions": suggestions[:3],  # Limit to top 3 suggestions
                    }
                )

    # Calculate score based on percentage of correctly spelled words
    total_words = len(words)
    correct_words = total_words - len(misspelled)
    score = (correct_words / total_words * 100) if total_words > 0 else 100

    # Cap at 100 and convert to scale of 10
    score = min(100, score) / 10

    return score, misspelled


def find_closest_words(word, word_set, max_distance=2):
    """Find closest matching words for spelling correction"""
    return [w for w in word_set if edit_distance(word, w) <= max_distance][:5]


def check_grammar(text):
    """Check for grammar errors in the CV text"""
    # Try to import language_tool_python or use simplified grammar check
    try:
        import language_tool_python

        print("  Using LanguageTool for grammar checking...")

        # Initialize LanguageTool for grammar checking
        try:
            tool = language_tool_python.LanguageTool("en-US")
            print("  LanguageTool initialized successfully")

            # Check the text
            print("  Checking grammar...")
            matches = tool.check(text)
            print(f"  Found {len(matches)} potential issues")

            # Extract grammar issues
            grammar_issues = []
            for match in matches:
                # Filter out spelling mistakes which are handled separately
                if match.ruleId != "MORFOLOGIK_RULE_EN_US":
                    grammar_issues.append(
                        {
                            "message": match.message,
                            "context": text[
                                max(0, match.offset - 20) : match.offset
                                + match.length
                                + 20
                            ],
                            "suggestion": (
                                match.replacements[0] if match.replacements else ""
                            ),
                        }
                    )

            # Calculate score based on number of issues per 100 words
            words = text.split()
            word_count = len(words)

            # Score formula: 10 - (issues per 100 words), capped between 0-10
            issue_ratio = (
                len(grammar_issues) / (word_count / 100) if word_count > 0 else 0
            )
            score = max(0, min(10, 10 - issue_ratio))
            print(
                f"  Grammar check complete with LanguageTool, found {len(grammar_issues)} grammar issues"
            )

        except Exception as e:
            # Fallback if LanguageTool initialization fails
            print(f"  LanguageTool initialization failed: {str(e)}")
            print("  Using simplified grammar checking instead.")
            score, grammar_issues = simplified_grammar_check(text)

    except ImportError:
        # Fallback if LanguageTool is not available
        print("  language_tool_python not available, using simplified grammar checking")
        score, grammar_issues = simplified_grammar_check(text)

    return score, grammar_issues


def simplified_grammar_check(text):
    """Simplified grammar check when LanguageTool is not available"""
    print("  Running simplified grammar check...")
    grammar_issues = []

    # Simple sentence splitting fallback
    print("  Splitting text into sentences...")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    print(f"  Found {len(sentences)} sentences")

    # Simple checks
    print("  Checking for grammar issues...")
    for sentence in sentences:
        # Check for repeated words
        words = re.findall(r"\b\w+\b", sentence.lower())
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower() and words[i].isalpha():
                grammar_issues.append(
                    {
                        "message": f"Repeated word: '{words[i]}'",
                        "context": sentence,
                        "suggestion": f"Remove one instance of '{words[i]}'",
                    }
                )

        # Check for missing periods at the end of sentences
        if sentence and not sentence[-1] in [".", "!", "?", ":"]:
            grammar_issues.append(
                {
                    "message": "Sentence doesn't end with proper punctuation",
                    "context": sentence,
                    "suggestion": "Add a period or appropriate punctuation",
                }
            )

    # Calculate approximate score
    score = max(0, 10 - len(grammar_issues) * 0.5)
    print(f"  Simplified grammar check complete, found {len(grammar_issues)} issues")

    return score, grammar_issues


def check_formatting_consistency(cv_data):
    """Check for formatting inconsistencies in the CV"""
    issues = []

    # Check section header consistency
    sections = cv_data.get("sections", [])
    if sections:
        # Check if section headers have consistent formatting
        section_fonts = {}
        for section in sections:
            if section["type"] == "main":
                key = f"{section['is_bold']}_{section['font_size']}"
                section_fonts[key] = section_fonts.get(key, 0) + 1

        total_sections = len([s for s in sections if s["type"] == "main"])
        if total_sections > 1:
            most_common_format = max(section_fonts.items(), key=lambda x: x[1])
            consistency_ratio = most_common_format[1] / total_sections

            if consistency_ratio < 0.8:
                issues.append("Section headers have inconsistent formatting")

    # Check bullet point consistency
    lists = cv_data.get("lists", [])
    if lists:
        bullet_styles = {}
        for lst in lists:
            for item in lst.get("items", []):
                bullet_text = item.get("text", "")
                if bullet_text:
                    bullet_char = (
                        bullet_text[0]
                        if bullet_text[0] in ["•", "-", "✓", "✔", "➢", "*"]
                        else "other"
                    )
                    bullet_styles[bullet_char] = bullet_styles.get(bullet_char, 0) + 1

        if len(bullet_styles) > 1:
            issues.append(
                "Inconsistent bullet point styles used throughout the document"
            )

    # Check paragraph alignment consistency
    alignment_metrics = cv_data.get("format_features", {}).get("alignment_metrics", {})
    alignment_consistency = alignment_metrics.get("alignment_consistency", 0)

    if alignment_consistency < 0.85:
        issues.append("Text alignment is inconsistent throughout the document")

    # Calculate score based on issues
    base_score = 10
    deduction_per_issue = 1.5
    score = max(0, base_score - (len(issues) * deduction_per_issue))

    # Bonus for excellent consistency
    if alignment_consistency > 0.95 and not issues:
        score = min(10, score + 1)

    return score, issues


def evaluate_fonts(cv_data):
    """Evaluate font choices and consistency in the CV"""
    issues = []

    # Get font data
    fonts = cv_data.get("fonts", [])
    font_metrics = cv_data.get("format_features", {}).get("font_metrics", {})

    # Check number of fonts used
    unique_font_count = font_metrics.get("unique_font_count", 0)
    if unique_font_count > 4:
        issues.append(
            f"Too many different fonts used ({unique_font_count}). Limit to 2-3 for professional appearance."
        )

    # Check font names for professionalism
    unprofessional_fonts = ["Comic Sans", "Papyrus", "Curlz", "Jokerman", "Chiller"]
    for font in fonts:
        font_name = font.get("name", "").lower()
        for unprofessional in unprofessional_fonts:
            if unprofessional.lower() in font_name:
                issues.append(f"Unprofessional font detected: {font.get('name')}")

    # Check for extremely small or large font sizes
    for font in fonts:
        size = font.get("size", 0)
        if size < 8:
            issues.append(f"Font size too small: {size}pt")
        elif size > 36 and font.get("count", 0) > 5:  # Allow large fonts for titles
            issues.append(f"Font size too large: {size}pt")

    # Check primary font ratio
    primary_font_ratio = font_metrics.get("primary_font_ratio", 0)
    if primary_font_ratio < 0.6:
        issues.append(
            "No consistent primary font. One font should be used for most text."
        )

    # Calculate score based on issues and metrics
    base_score = 10

    # Deduct for issues
    deduction_per_issue = 1.5
    issue_deduction = min(8, len(issues) * deduction_per_issue)

    # Adjust based on font metrics
    font_variety_score = (
        font_metrics.get("font_variety_score", 0) * 3
    )  # Scale to 0-3 range

    # Final score calculation
    score = max(0, min(10, base_score - issue_deduction + font_variety_score))

    return score, issues


def analyze_spacing(cv_data):
    """Analyze spacing and layout issues in the CV"""
    issues = []

    # Get spacing metrics
    spacing_metrics = cv_data.get("format_features", {}).get("spacing_metrics", {})
    density_metrics = cv_data.get("format_features", {}).get("density_metrics", {})

    # Check line spacing consistency
    line_spacing_consistency = spacing_metrics.get("line_spacing_consistency", 0)
    if line_spacing_consistency < 0.7:
        issues.append("Inconsistent line spacing throughout the document")

    # Check line height consistency
    line_height_consistency = spacing_metrics.get("line_height_consistency", 0)
    if line_height_consistency < 0.7:
        issues.append("Inconsistent text size or line heights")

    # Check for double spacing issues
    line_spacing_avg = spacing_metrics.get("line_spacing_avg", 0)
    line_height_avg = spacing_metrics.get("line_height_avg", 0)

    if line_spacing_avg > 2 * line_height_avg:
        issues.append("Excessive spacing between lines (double spacing)")

    # Check text density
    text_density = density_metrics.get("text_density", 0)
    whitespace_ratio = density_metrics.get("whitespace_ratio", 0)

    if text_density > 0.7:
        issues.append("Text is too dense. Add more white space for better readability.")
    elif whitespace_ratio > 0.7:
        issues.append("Too much white space. Document appears too sparse.")

    # Check margins through left alignment analysis
    alignment_metrics = cv_data.get("format_features", {}).get("alignment_metrics", {})
    left_margin_std = alignment_metrics.get("left_margin_std", 0)

    if left_margin_std > 30:  # High standard deviation in left margins
        issues.append("Inconsistent margins throughout the document")

    # Calculate score based on issues and metrics
    base_score = 10

    # Deduct for issues
    deduction_per_issue = 1.5
    issue_deduction = min(8, len(issues) * deduction_per_issue)

    # Bonus for good spacing
    spacing_bonus = 0
    if line_spacing_consistency > 0.9 and text_density > 0.3 and text_density < 0.6:
        spacing_bonus = 1

    # Final score calculation
    score = max(0, min(10, base_score - issue_deduction + spacing_bonus))

    return score, issues


def analyze_content_quality(text, lines):
    """Analyze the quality and professionalism of CV content"""
    issues = []

    # Check for consistent tense usage
    tense_issues = check_tense_consistency(lines)
    issues.extend(tense_issues)

    # Check for business language and professionalism
    professionalism_issues = check_professionalism(text)
    issues.extend(professionalism_issues)

    # Check for actionable language
    action_verbs = [
        "achieved",
        "managed",
        "created",
        "developed",
        "implemented",
        "increased",
        "decreased",
        "improved",
        "negotiated",
        "coordinated",
        "led",
        "directed",
        "organized",
        "produced",
        "supervised",
    ]

    contains_action_verbs = False
    for verb in action_verbs:
        if verb in text.lower():
            contains_action_verbs = True
            break

    if not contains_action_verbs:
        issues.append(
            "Use more action verbs to describe achievements and responsibilities"
        )

    # Check sentence structure variety
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 5:  # Only check if there are enough sentences
        sentence_beginnings = [
            s.split()[0].lower() if s.split() else "" for s in sentences
        ]
        beginning_counts = Counter(sentence_beginnings)

        # Check if more than 30% of sentences start with the same word
        for word, count in beginning_counts.items():
            if word and count / len(sentences) > 0.3:
                issues.append(
                    f"Too many sentences begin with '{word}'. Vary sentence structure."
                )

    # Calculate score based on issues
    base_score = 10
    deduction_per_issue = 1.2
    score = max(0, base_score - (len(issues) * deduction_per_issue))

    return score, issues


def check_tense_consistency(lines):
    """Check for consistency in verb tense, especially in bullet points"""
    issues = []

    # Look for bullet points that likely contain achievements or responsibilities
    bullet_lines = [
        line
        for line in lines
        if line.strip().startswith(("•", "-", "✓", "✔", "➢", "*"))
        or (len(line) > 2 and line[0].isdigit() and line[1] == ".")
    ]

    # Common past tense verbs in resumes
    past_tense_verbs = [
        "managed",
        "developed",
        "created",
        "achieved",
        "implemented",
        "increased",
        "decreased",
        "improved",
        "led",
        "coordinated",
    ]

    # Common present tense verbs in resumes
    present_tense_verbs = [
        "manage",
        "develop",
        "create",
        "achieve",
        "implement",
        "increase",
        "decrease",
        "improve",
        "lead",
        "coordinate",
    ]

    past_tense_count = 0
    present_tense_count = 0

    for line in bullet_lines:
        words = line.lower().split()

        if words:
            # Check if line starts with past tense
            for verb in past_tense_verbs:
                if verb in words[:3]:  # Check first few words
                    past_tense_count += 1
                    break

            # Check if line starts with present tense
            for verb in present_tense_verbs:
                if verb in words[:3]:  # Check first few words
                    present_tense_count += 1
                    break

    # Determine if there's inconsistency
    if past_tense_count > 0 and present_tense_count > 0:
        # If there's a mix but one tense dominates (>80%), suggest standardizing
        total = past_tense_count + present_tense_count
        if total > 3:  # Only check if there are enough verb instances
            if 0 < past_tense_count / total < 0.2:
                issues.append(
                    "Inconsistent verb tense: Standardize to present tense throughout"
                )
            elif 0 < present_tense_count / total < 0.2:
                issues.append(
                    "Inconsistent verb tense: Standardize to past tense throughout"
                )
            else:
                issues.append(
                    "Inconsistent verb tense: Mix of past and present tense verbs"
                )

    return issues


def check_professionalism(text):
    """Check for unprofessional language or constructs"""
    issues = []

    # Check for first person pronouns (generally avoided in CVs)
    first_person = [
        "I ",
        "I've",
        "I'm",
        "I'll",
        "I'd",
        "me ",
        "my ",
        "mine ",
        "myself ",
    ]
    for pronoun in first_person:
        if pronoun.lower() in text.lower():
            issues.append("Avoid first-person pronouns in professional CVs")
            break

    # Check for informal language
    informal_terms = [
        "stuff",
        "things",
        "guy",
        "awesome",
        "cool",
        "kind of",
        "sort of",
        "a lot",
        "really",
    ]
    for term in informal_terms:
        if f" {term} " in f" {text.lower()} ":  # Add spaces to match whole words
            issues.append(
                f"Informal language detected: '{term}'. Use more professional terminology."
            )
            break

    # Check for clichés and overused phrases
    cliches = [
        "team player",
        "detail-oriented",
        "hard worker",
        "motivated",
        "thinking outside the box",
        "go-getter",
        "proactive",
        "self-starter",
    ]

    for cliche in cliches:
        if cliche.lower() in text.lower():
            issues.append(
                f"Cliché detected: '{cliche}'. Replace with specific achievements."
            )
            break

    return issues


def generate_overall_feedback(scores):
    """Generate overall feedback based on scores"""
    feedback = []

    # Add category-specific feedback
    categories = [
        "spelling",
        "grammar",
        "formatting",
        "fonts",
        "spacing",
        "content_quality",
    ]

    for category in categories:
        score = scores[category]["score"]
        issues = scores[category]["issues"]

        if score < 5:
            if category == "spelling":
                feedback.append(
                    f"Critical spelling issues need attention. {len(issues)} errors found."
                )
            elif category == "grammar":
                feedback.append(
                    f"Significant grammar problems detected. {len(issues)} issues found."
                )
            elif category == "formatting":
                feedback.append(
                    "Poor formatting consistency makes the CV look unprofessional."
                )
            elif category == "fonts":
                feedback.append("Font choices and usage need significant improvement.")
            elif category == "spacing":
                feedback.append(
                    "Layout and spacing issues make the document hard to read."
                )
            elif category == "content_quality":
                feedback.append(
                    "Content quality needs substantial improvement for professional impact."
                )
        elif score < 7:
            if issues:
                feedback.append(
                    f"{category.replace('_', ' ').title()} needs improvement: {len(issues)} issues found."
                )

    # Add overall assessment
    overall_score = scores["overall"]
    if overall_score < 5:
        feedback.insert(
            0,
            f"Overall CV quality is poor ({overall_score}/10). Major revisions needed.",
        )
    elif overall_score < 7:
        feedback.insert(
            0,
            f"CV quality is average ({overall_score}/10). Several improvements recommended.",
        )
    elif overall_score < 8.5:
        feedback.insert(
            0, f"CV quality is good ({overall_score}/10). Minor improvements suggested."
        )
    else:
        feedback.insert(
            0,
            f"Excellent CV quality ({overall_score}/10). Only minor refinements suggested.",
        )

    # Update the scores dictionary with feedback
    scores["feedback"] = feedback


if __name__ == "__main__":
    # Path to the JSON file created by step1.py
    json_path = "cv_format_data.json"

    # Analyze CV quality
    results = analyze_cv_quality(json_path)

    # Display results
    print("\n" + "=" * 50)
    print("CV QUALITY ANALYSIS RESULTS")
    print("=" * 50)

    print(f"\nOverall ATS Score: {results['overall']}/10")
    print("\nCategory Scores:")
    print(f"Spelling: {results['spelling']['score']}/10")
    print(f"Grammar: {results['grammar']['score']}/10")
    print(f"Formatting: {results['formatting']['score']}/10")
    print(f"Fonts: {results['fonts']['score']}/10")
    print(f"Spacing: {results['spacing']['score']}/10")
    print(f"Content Quality: {results['content_quality']['score']}/10")

    print("\nKey Issues:")
    for category in [
        "spelling",
        "grammar",
        "formatting",
        "fonts",
        "spacing",
        "content_quality",
    ]:
        issues = results[category]["issues"]
        if issues:
            print(f"\n{category.replace('_', ' ').title()} Issues:")
            if category == "spelling" or category == "grammar":
                # Limit to first 5 issues for these categories
                for issue in issues[:5]:
                    if category == "spelling":
                        print(
                            f"- Misspelled: '{issue['word']}' (Suggestions: {', '.join(issue['suggestions'])})"
                        )
                    else:
                        print(f"- {issue['message']}")
                if len(issues) > 5:
                    print(f"  ...and {len(issues) - 5} more issues")
            else:
                for issue in issues:
                    print(f"- {issue}")

    print("\nFeedback Summary:")
    for feedback in results["feedback"]:
        print(f"- {feedback}")

    # Save results to JSON for further analysis
    with open("cv_quality_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nDetailed analysis saved to cv_quality_analysis.json")
