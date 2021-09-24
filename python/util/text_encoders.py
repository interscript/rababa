
#from util import text_cleaners
from typing import Dict, List, Optional

from util_nakdimon import nakdimon_dataset as dataset
from util_nakdimon import nakdimon_hebrew_model as hebrew


class TextEncoder:
    pad = "P"

    def __init__(
        self,
        cleaner_fn: Optional[str] = None,
        config = None, #: Dict[str] = None,
    ):
        if cleaner_fn:
            self.cleaner_fn = getattr(text_cleaners, cleaner_fn)
        else:
            self.cleaner_fn = None

    def str_to_data(self, text: str, heb_items=[]) -> dataset.Data:
        return dataset.from_text(text, self.config['max_len'])

    def data_to_str(self, data: dataset.Data, diacritics=None):
        if diacritics is None:
            return 'not implemented yet'
        else:
            return 'not implemented yet'

    def clean(self, text):
        if self.cleaner_fn:
            return self.cleaner_fn(text)
        return text

    def __str__(self):
        return type(self).__name__


# class HebrewEncoder(TextEncoder):
