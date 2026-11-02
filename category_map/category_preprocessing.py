import re

from category_config import CATEGORIES
from category_keywords import KEYWORDS


IRREGULAR = {
    'buses': 'bus', 'minibuses': 'minibus', 'gases': 'gas', 'lenses': 'lens', 'canvases': 'canvas',
    'tvs': 'tv', 'cds': 'cd', 'dvds': 'dvd', 'cvs': 'cv', 'djs': 'dj', 'rvs': 'rv', 'pcs': 'pc',
    'movies': 'movie', 'hoodies': 'hoodie', 'onesies': 'onesie', 'neckties': 'necktie',
    'beanies': 'beanie', 'cookies': 'cookie', 'smoothies': 'smoothie', 'walkies': 'walkie',
    'talkies': 'talkie', 'lingeries': 'lingerie',
    'shelves': 'shelf', 'scarves': 'scarf', 'calves': 'calf', 'knives': 'knife', 'leaves': 'leaf',
    'loaves': 'loaf', 'halves': 'half',
    'tomatoes': 'tomato', 'potatoes': 'potato', 'mangoes': 'mango',
    'children': 'child', 'mice': 'mouse', 'feet': 'foot', 'teeth': 'tooth',
}

def flatten_category_hierarchy(categories=CATEGORIES):
    '''
    Returns (all_paths, flat_map)
    flat_map:  {'real estate > long term rentals': 'Real Estate', ...}
    all_paths: ['real estate > long term rentals', ...]   (lowercase 'main > sub')
    The sub category name alone is not unique ("Accessories", "Everything Else"), so the path is used
    '''
    flat_map = {}
    all_paths = []

    for main_cat, subs in categories.items():
        for sub_cat in subs:
            path = f'{main_cat} > {sub_cat}'.lower()
            flat_map[path] = main_cat
            all_paths.append(path)

    return all_paths, flat_map


def singular(word):
    '''
    Simple plural -> singular: accessories -> accessory, watches -> watch, jeans -> jean
    '''

    EXCEPTIONS = {
    'plus', 'bonus', 'status', 'campus', 'focus', 'virus', 'cactus', 'citrus', 'genius',
    'canvas', 'news', 'series', 'species', 'chassis', 'atlas', 'bias', 'iris', 'tennis',
    'christmas', 'lens', 'lexus', 'asparagus', 'hummus', 'couscous',
    }

    if len(word) <= 3 or word in EXCEPTIONS:
        return word
    if word.endswith('ies'):
        return word[:-3] + 'y' # accessories -> accessory
    if word.endswith(('ches', 'shes', 'xes', 'sses')):
        return word[:-2]  # watches -> watch, dresses -> dress
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def clean(text):
    text = str(text).lower().replace("'", '').replace('\u2019', '')   # Children's -> childrens
    words = re.findall(r'\w+', text)
    return ' '.join(IRREGULAR.get(w) or singular(w) for w in words)


# Clean the keywords once, not on every call
# A keyword with more words gives more points ("power tool" = 2, "tool" = 1)
CLEAN_KEYWORDS = {path: [(clean(kw), len(kw.split())) for kw in keywords]for path, keywords in KEYWORDS.items()}


def classify(item):
    """
    Returns 'Main category > Sub category', 'Main category' or 'Other'.

    Every keyword found in the text adds points to its sub category and the sub
    category with the most points wins.
    If several sub categories of the SAME main category tie (for example "apartment"
    fits all Real Estate subs), only the main category is returned.
    If nothing matches, the result is 'Other'.
    """
    if item is None or item != item: # None or NaN
        return 'Other'

    text = ' ' + clean(item) + ' '  

    scores = {}
    for path, keywords in CLEAN_KEYWORDS.items():
        points = sum(p for kw, p in keywords if f' {kw} ' in text)
        if points:
            scores[path] = points

    if not scores:
        return 'Other'

    best = max(scores.values())
    winners = [path for path, points in scores.items() if points == best]

    mains = {path.split(' > ')[0] for path in winners}
    if len(winners) == 1:
        return winners[0]
    if len(mains) == 1:
        return mains.pop()
    return winners[0]


if __name__ == '__main__':
    for sample in ['iphone 13', 'cartier wristwatch', 'baby monitor', 'toyota camry 2015', '3 room apartment in yerevan', 'something strange']:
        print(f'{sample!r:32} -> {classify(sample)}')






                                                              