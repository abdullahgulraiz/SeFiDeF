from typing import Collection, Union, Tuple
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


def remove_stopwords(input_s: Union[Collection[str], str], tokenize: bool = False) -> Union[Tuple[str], str]:
    def _remove_stopwords(list_of_words: Collection[str]) -> str:
        result = [word.lower() for word in list_of_words if word not in stop_words]
        return " ".join(result) if not tokenize else result
    if isinstance(input_s, str):
        return _remove_stopwords(input_s.split(" "))
    else:
        return tuple([_remove_stopwords(temp.split(" ")) for temp in input_s])
