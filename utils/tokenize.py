import re
from typing import List

from tokenizers import pre_tokenizers
from tokenizers.normalizers import *
from tokenizers.pre_tokenizers import *


# todo : need refine steps(e.g. 't', 'ss')
def normalize(statement: str) -> str:
    return NFD().normalize_str(statement)


def pre_tokenize(statement: str) -> List[str]:
    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=False)])
    res = pre_tokenizer.pre_tokenize_str(statement)
    res_tokens = [t[0] for t in res]
    return res_tokens


def camel_case_split(identifier:str) -> List[str]:
    # using regex to match camel cases
    return re.findall(r'[a-z]+|[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', identifier)


def post_tokenize(tokens:List[str]) -> List[str]:
    res = []
    for t in tokens:
        tt = re.sub(r'[^a-zA-Z]', '', t)
        if len(tt) > 0:
           res.append(tt.lower())
    return res


def tokenize(statement: str, specials: List[str]):
    if statement in specials:
        return [statement]
    res = []
    ns = normalize(statement)
    identifiers = pre_tokenize(ns)
    for id in identifiers:
        tokens = camel_case_split(id)
        refined_tokens = post_tokenize(tokens)
        res.extend(refined_tokens)
    return res
