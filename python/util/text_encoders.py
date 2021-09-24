
from util import text_cleaners
from typing import Dict, List, Optional

from util_nakdimon import nakdimon_dataset as dataset
from util_nakdimon import nakdimon_hebrew_model as hebrew


class TextEncoder:

    def __init__(
        self,
        config = None, # Dict[str, Any] = None,
    ):
        self.config = config
        if self.config.get('text_cleaner', False):
            self.cleaner_fn = getattr(text_cleaners,
                                      self.config['text_cleaner'])
        else:
            self.cleaner_fn = None

    def str_to_data(self, text: str) -> dataset.Data: #, heb_items=[]) -> dataset.Data:
        return dataset.Data.from_text([text], self.config['max_len'])

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


class HebraicEncoder(TextEncoder):
    pass
