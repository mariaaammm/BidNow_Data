# Category map

Finds the category of a listing title, for example `iphone 13` -> `Electronics > Phones and Tablets`.

There are two parts:

1. **Rule-based classifier** (`classify()`): keywords and points. Fast and easy to edit.
2. **NLP baseline** (`v2/`): a machine learning model trained on synthetic titles. First version, to be retrained on real titles.

## Files

| File | What it is |
|---|---|
| `categories.json` | Category tree (main > sub > leaf, 100 sub categories) |
| `category_keywords.json` | Keywords per sub category (lowercase and singular) |
| `category_preprocessing.py` | `clean()` and `classify()` |
| `test_category_preprocessing.py` | 53 test cases for `classify()` |
| `v2/generate_data.py` | Makes fake labelled titles for the NLP model |
| `v2/train_category_model.py` | Trains the NLP model, prints the scores, predicts categories |
| `data/category_data.csv` | Created by `generate_data.py` (not in git) |
| `v2/category_model.joblib` | Created by `train_category_model.py` (not in git) |

## Setup

From the project root, with the virtual environment active (see the main README):

```bash
pip install -r requirements.txt
pip install scikit-learn joblib      # only for the NLP baseline; not in requirements.txt yet
```

## 1. Rule-based classifier

```python
from category_preprocessing import classify

classify('iphone 13')            # 'Electronics > Phones and Tablets'
classify('3 room apartment')     # 'Real Estate'
classify('qwerty')               # 'Other'
```

Every keyword found in a title gives points to its sub category and a longer phrase gives more points ("power tool" = 2, "tool" = 1). The sub category with the most points wins. A tie inside one main category returns only the main category. No match returns `Other`.

To change categories or keywords, edit the JSON files and run the tests:

```bash
cd category_map
pytest -q
```

## 2. NLP baseline

```bash
python category_map/v2/generate_data.py                  # writes category_map/data/category_data.csv
python category_map/v2/train_category_model.py train     # trains, prints scores, saves category_model.joblib
python category_map/v2/train_category_model.py predict "toyota camry 2015 yerevan" "iphon 13 pro unlocked"
```

`predict` prints the 3 most likely categories with their probability:

```
'iphon 13 pro unlocked'
   57.1%  Electronics > Phones and Tablets
    4.4%  Electronics > Laptops and Computers
    4.3%  Electronics > Photo and Video Equipment
```

### Training data (`generate_data.py`)

Builds fake titles from the category tree and the keywords. One keyword or leaf name is the base of each title, for example `used toyota camry 2015 Yerevan` with the label `Vehicles > Cars`. Around it the script adds:

- a condition, a colour and details that fit the main category (size, year, number of rooms, salary and so on) and a city
- short phrases such as "negotiable" or "delivery available"
- Armenian words written in Latin letters (`lav vijakum`, `sev`, `shtap`) among these filler words
- a typo in some titles (about 12%)

Options: `--per-sub` (titles per sub category, default 150), `--seed` (same seed gives the same file, default 42), `--out`, `--map-dir`.

Output columns: `title`, `category` (`Main > Sub`), `split` (`train` or `test`). Sub categories with no keywords and no leaf names (such as the empty "Everything Else" ones) get no titles, so the data covers 95 of the 100 sub categories.

**Train and test split.** For a sub category with at least 4 base words, 20% of the base words are held out and used only for test titles. So the test score shows how the model handles words it has not seen. Sub categories with fewer base words are split at random, so train and test share words there.

### Model (`train_category_model.py`)

TF-IDF on word pairs (meaning) and on character groups of 2 to 5 letters (typos such as `iphon`), then logistic regression. `train` also prints the rule-based `classify()` score on the same test titles.

To train on your own data, use a CSV with the columns `title` and `category` (and optionally `split`):

```bash
python category_map/v2/train_category_model.py train --data path/to/real_titles.csv
```

Without a `split` column with both `train` and `test` values, the script splits the rows 80/20 at random.

### Result

Default seed 42: 11,041 training titles, 2,748 test titles, 95 categories. Exact sub category on the test titles:

| | Exact accuracy | Macro F1 |
|---|---|---|
| Rule-based `classify()` | 77.4% | n/a |
| NLP baseline | 46.9% | 0.441 |

Your numbers can differ a little if the data file was made with another seed or another version of the generator.

**How to read this**

- The rule-based number is not a fair comparison. The test titles are built from its own keyword list, so it knows every test word. The NLP model does not. When `classify()` finds only a main category or `Other`, it counts as wrong here.
- The test is built to show how the model handles unseen words. The keyword rules cannot handle a word that is not in the list. The model can still guess from similar letters and words.
- Some keywords belong to several sub categories (`apartment` is in four Real Estate sub categories). The generator uses them for each one, so similar titles get different labels. This probably limits the accuracy of any model trained on this file.
- All titles are synthetic and built from our own keywords, so every number here is optimistic. The real accuracy is not measured yet.

## Limits

- Synthetic training data. Replace or extend it with real BidNow or list.am titles (a few hundred for each main category) and retrain.
- The word that defines the category is always English. Armenian appears only in the filler words, written in Latin letters. No Armenian script and no Russian titles.
- The NLP model has no tests yet.
- Do not commit generated files. `category_map/data/` and `*.joblib` are in `.gitignore`.