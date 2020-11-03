import numpy as np
from numpy import sqrt, abs, round, square, mean
from scipy.stats import norm

def get_attribute_words():
    with open('words/final_pleasant_words', 'r') as f:
        pleasant_words = f.read().split('\n')
    with open('words/final_unpleasant_words', 'r') as f:
        unpleasant_words = f.read().split('\n')
    with open('words/career_words', 'r') as f:
        career_words = f.read().split('\n')
    with open('words/final_family_words', 'r') as f:
        family_words = f.read().split('\n')

    return pleasant_words, unpleasant_words, career_words, family_words

def contain_word(text, set):
    if any([True if w in set else False for w in text.split()]):
        return True
    else:
        return False


def remove_repeative_redundant_puncts(string):
    str_list = string.split()
    new_list = []
    for t in str_list:
        if len(new_list) > 0:
            if t in [',', '.', '!', '?', '*', '\'', '"'] and t == new_list[-1]:
                continue
            else:
                new_list.append(t)
        else:
            new_list.append(t)

    return ' '.join(new_list)

def z_test(x1, x2):
    n1 = len(x1)
    n2 = len(x2)

    x1 = np.array(x1, dtype=np.float32)
    x2 = np.array(x2, dtype=np.float32)
    x1_mean = mean(x1)
    x2_mean = mean(x2)

    S1 = np.sum(square(x1 - x1_mean)) / (n1 - 1)
    S2 = np.sum(square(x2 - x2_mean)) / (n2 - 1)

    numerator = x1_mean - x2_mean
    denominator = sqrt((S1 / n1) + (S2 / n2))

    # print(numerator, denominator)

    z = numerator / denominator

    p = 1 - norm.cdf(abs(z))

    return round(z, 3), p