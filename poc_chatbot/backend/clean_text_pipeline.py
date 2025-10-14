import unicodedata
import re
from html import unescape


class CleanTextPipeline:
    def __init__(self):
        self._re_crlf = re.compile(r"\r\n?")
        self._re_many_blank = re.compile(r"\n{3,}")
        self._re_single_newline = re.compile(r"(?<!\n)\n(?!\n)")
        self._re_multi_space = re.compile(r"[ \t]{2,}")
        self._re_space_around_newline = re.compile(r" *\n *")
        self._re_dehyphen = re.compile(r"(\w)-\s*\n\s*(\w)")
        self._re_control = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
        self._re_html_tags = re.compile(r"<[^>]+>")
        self._re_url = re.compile(r"https?://\S+|www\.\S+")
        self._re_email = re.compile(r"\b[\w.-]+?@\w+?\.\w+?\b")
        self._re_space_punct = re.compile(r"\s+([,.;:!?])")
        self._re_repeat_punct = re.compile(r"([!?.,]){2,}")
        self._re_trailing_spaces = re.compile(r"[ \t]+$|\A[ \t]+", re.M)
        self._re_sentence_end_double_newline = re.compile(r"([.!?])\s*\n\n")
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize data with Normalization Form Compatibility Decomposition"""
        return unicodedata.normalize("NFKC", text)

    def dehyphenate(self, text: str) -> str:
        """Join words split across lines with hyphenation, e.g. 'ex-\nample' -> 'example'"""
        return self._re_dehyphen.sub(r"\1\2", text)

    def remove_control_characters(self, text: str) -> str:
        """Remove non-printable/control characters that may break downstream tools."""
        return self._re_control.sub("", text)

    def strip_html(self, text: str) -> str:
        """Remove simple HTML tags and unescape entities."""
        text = self._re_html_tags.sub("", text)
        return unescape(text)
    
    def remove_mid_sentence_double_newlines(self, text: str) -> str:
        """Remove \n\n that are not at the end of a sentence (e.g., after ., !, or ?). 
        Keeps \n\n for paragraph breaks; replaces mid-sentence \n\n with a single \n."""
        sentence_placeholder = "@@SENTENCE_DOUBLE@@"
        # Step 1: Replace sentence-ending \n\n with a unique placeholder
        text = self._re_sentence_end_double_newline.sub(sentence_placeholder, text)
        # Step 2: Replace all remaining \n\n (mid-sentence) with \n
        text = text.replace("\n\n", " ")
        # Step 3: Restore sentence-ending placeholders back to \n\n
        text = text.replace(sentence_placeholder, "\n\n")
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace: line endings, collapse many blank lines, convert single newlines to space, trim."""
        text = self._re_crlf.sub("\n", text)
        text = self._re_many_blank.sub("\n\n", text)
        text = self._re_single_newline.sub(" ", text)
        text = self._re_multi_space.sub(" ", text)
        text = self._re_space_around_newline.sub("\n", text)
        text = self._re_trailing_spaces.sub("", text)
        return text.strip()

    def fix_punctuation_spacing(self, text: str) -> str:
        """Remove space before punctuation and collapse repeated punctuation runs."""
        text = self._re_space_punct.sub(r"\1", text)
        # collapse repeated punctuation like "!!!" -> "!"
        text = self._re_repeat_punct.sub(r"\1", text)
        return text

    def collapse_newlines(self, text: str) -> str:
        """Backward compatible alias to keep existing behavior if used externally."""
        return self.normalize_whitespace(text)

    def run(self, text: str) -> str:
        """Run cleaning pipeline. Pass steps list to customize order. Default is safe common sequence."""
        if not isinstance(text, str):
            raise TypeError("text must be a str")

        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        text = self.strip_html(text)
        text = self.dehyphenate(text)
        text = self.remove_mid_sentence_double_newlines(text)
        text = self.normalize_whitespace(text)
        text = self.fix_punctuation_spacing(text)


        return text


# if __name__ == "__main__":
#     clean_text_pipeline = CleanTextPipeline()
#     example_text = """This is\n\n\n\n an ex-ample text with hyphenation \n\n\n\n at the end of a line.
# It also contains multiple
# newlines and    irregular   spacing.
# Additionally, it has some unicode characters like ﬁ and ﬂ.
# Visit https://example.com or mail me@domain.com. <b>bold</b>"""
#     cleaned_text = clean_text_pipeline.run(example_text)
#     print("Original Text:")
#     print(example_text)
#     print("\nCleaned Text:")
#     print(cleaned_text)
