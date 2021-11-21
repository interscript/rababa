from typing import Dict, List, Optional

# from util import text_cleaners

from util import nakdimon_dataset as dataset
from util import nakdimon_hebrew_model as hebrew


class TextEncoder:
    def __init__(
        self, config=None,  # Dict[str, Any] = None,
    ):
        self.config = config

    def str_to_data(self, text: str) -> dataset.Data:
        return dataset.Data.from_text([text], self.config["max_len"])

    def data_to_str(self, data: dataset.Data, diacritics=None):
        if diacritics is None:
            return "not implemented yet"
        else:
            return "not implemented yet"

    def clean(self, text):
        if self.cleaner_fn:
            return self.cleaner_fn(text)
        return text

    def __str__(self):
        return type(self).__name__


class HebraicEncoder(TextEncoder):
    pass
