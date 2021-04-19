#!/usr/bin/env python
__author__ = 'ziyan.yin'

from copy import deepcopy
from typing import Dict, List

import stringutils
import numpy as np


class HotWordsRecognize:
    """
        input articles, comments, issues or files
            >>> this = HotWordsRecognize()
            >>> this.read_file('z:/comments.txt')
        or
            >>> this.add_content('you are my best')
        use recognize method to analyze total content
            >>> words = this.recognize(top=10)
        words like [('a', 1000), ('b', 200)] in times order desc
    """
    __slots__ = ['content', 'keywords', 'word_tree', 'similar']

    def __init__(self, similar=0.8):
        self.content: List[str] = []
        self.keywords: Dict[str, int] = dict()
        self.word_tree: Dict[str, Dict[str, int]] = dict()
        self.similar: float = similar

    def read_file(self, file, encoding='utf-8'):
        with open(file, 'r', encoding=encoding) as f:
            lines = f.readlines()
        for line in lines:
            self.add_content(line)

    def add_content(self, context: str):
        if not stringutils.is_empty(context):
            self.content.extend(stringutils.words_standard(context).split(' '))

    def _find_suffix_tree(self):
        def __suffix_words(head: str, words: str, length: int, tree: dict):
            if len(words) > length:
                if head not in tree:
                    tree[head] = dict()
                raw_word = words[:length + 1]
                if raw_word not in tree[head]:
                    tree[head][raw_word] = 0
                tree[head][raw_word] += 1

        for c in self.content:
            for i in range(len(c) - 1):
                for le in range(6):
                    __suffix_words(c[i], c[i + 1:], le, self.word_tree)

    def _filter_sparse_times(self):
        times = []
        for k, v in self.word_tree.items():
            for k_1, v_1 in v.items():
                times.append(v_1)
        median = np.median(times)
        for k, v in deepcopy(self.word_tree).items():
            for k_1, v_1 in v.items():
                if v_1 <= median + 1:
                    del self.word_tree[k][k_1]

    def _scan(self):
        for k, v in self.word_tree.items():
            word_list = sorted(v.items(), key=lambda x: x[0])
            parent = ('', 0)
            for word in word_list:
                if parent[0]:
                    if word[0].startswith(parent[0]):
                        if float(word[1])/float(parent[1]) <= self.similar:
                            self.keywords[k + parent[0]] = int(parent[1])
                    else:
                        self.keywords[k + parent[0]] = int(parent[1])
                parent = word
            if parent[0]:
                self.keywords[k + parent[0]] = int(parent[1])

    def _check_back(self, top=20, size=30):
        copy_keywords = sorted(self.keywords.items(), key=lambda x: x[1], reverse=True)[0:size]
        res = []
        for word, count in copy_keywords:
            if word in self.keywords:
                for ex_word in self.keywords:
                    if ex_word == word:
                        continue
                    if word in ex_word and self.keywords[ex_word]/count >= self.similar:
                        word = ex_word
                        count = self.keywords[ex_word]
                        break
            if (word, count) not in res:
                res.append((word, count))
        if len(res) < top:
            res = self._check_back(top, size+20)
        return sorted(res, key=lambda x: x[1], reverse=True)[:top]

    def recognize(self, top):
        self.word_tree.clear()
        if len(self.content) > 0:
            self._find_suffix_tree()
            self._filter_sparse_times()
            self._scan()
            return self._check_back(top, top+20)
        return []


def read_file(file):
    reg = HotWordsRecognize()
    reg.read_file(file)
    return reg.recognize(top=20)


def read_lines(lines):
    reg = HotWordsRecognize()
    for line in lines:
        reg.add_content(line)
    return reg.recognize(top=20)
