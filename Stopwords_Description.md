# Stopwords Description

## 1. Overview

The stopwords list for this project is defined in the file `bertopic_analysis/stopwords.py`, maintained as a Python list named `CUSTOM_STOPWORDS`. This list is designed specifically for **BERTopic topic modeling**, with the goal of filtering out high-frequency noise words that have no topic-discriminating power when clustering topics across 220 novels (20 Chinese web novels + 200 Western science fiction / fantasy novels), so that the extracted topic keywords more accurately reflect the core themes of transhumanism and science fiction literature.

### Basic Statistics

| Metric | Value |
|--------|-------|
| Total list entries | 3,101 |
| Unique words after deduplication | 2,722 |
| Duplicate entries | 379 |

> **Note**: Duplicate entries arise because the same word is listed separately under different category sections (for example, certain common names appear in both the character-name section and the general English stopwords section). `CountVectorizer` automatically deduplicates, so this does not affect the actual outcome.

---

## 2. File Structure

`stopwords.py` is divided into **6 major sections** by function, with each section separated by an `=====` comment line:

### Section 1: Chinese Web Novel Character Names (~82 words)

Covers the protagonist names, supporting-character names, and their pinyin forms across 20 Chinese web novels.

**Works covered**: A Thought Through Eternity, Battle Through the Heavens, Coiling Dragon, Cultivation Chat Group, Desolate Era, The Cultivation of the Four Myriad Years, I Shall Seal the Heavens, Super Mecha Genius, Library of Heaven's Path, Lord of the Mysteries, Monster Paradise, Pocket Hunter Dimension, Renegade Immortal, Reverend Insanity, Stellar Transformations, Swallowed Star, Tales of Demons and Gods, True Martial World, Versatile Mage, Warlock of the Magus World

### Section 2: Western Science Fiction / Fantasy Character Names (~524 words)

Covers character names across 200 Western science fiction and fantasy novels, grouped by series / author.

**Series covered** (partial examples):
- A Song of Ice and Fire
- Ender's Game
- Broken Earth Trilogy
- Dune
- Foundation
- Neuromancer
- The Three-Body Problem
- Plus more than 60 other standalone novels

### Section 3: General Noise Words (Largest Section)

Subdivided by part of speech into the following subcategories:

| Subcategory | Description | Examples |
|-------------|-------------|----------|
| Narrative verbs / dialogue tags | High-frequency narrative verbs in novels | said, asked, looked, walked |
| Supplementary speech verbs | Words used in dialogue scenes | muttered, exclaimed, snapped |
| Visual / observation verbs | Looking and gazing | glanced, peered, gazed |
| Physical / bodily actions | Bodily motions | grabbed, lifted, kicked |
| Body language / physiological reactions | Expressions and reactions | frowned, shrugged, trembled |
| Movement verbs | Locomotion | rushed, leaped, climbed |
| Psychological / cognitive verbs | Mental activities | realized, wondered, believed |
| High-frequency -ing gerunds | Present participle forms | feeling, walking, standing |
| Indefinite pronouns / -thing series | Generic pronouns | something, anyone, nowhere |
| Adverbs / intensifiers | Modifying adverbs | really, suddenly, extremely |
| High-frequency adverbs from Chinese-web-novel translations | Translation-specific terms | faintly, helplessly, instinctively |
| Dialogue fillers / interjections | Colloquial noise | oh, well, damn, hmm |
| High-frequency narrative nouns | Generic nouns | face, door, voice, moment |
| Body parts | Anatomical nouns | hand, eyes, shoulder, chest |
| Spatial / directional nouns | Scene nouns | ground, floor, wall, corner |
| Time nouns | Time expressions | seconds, morning, yesterday |
| Generic adjectives | High-frequency modifiers | old, dark, strong, strange |
| Emotional / mental-state adjectives | Emotional states | stunned, terrified, furious |
| Numerals | Quantity words | one, hundred, first, dozen |
| Sensitive / insulting vocabulary | Low-value offensive words | omitted |
| Publishing / e-book metadata | Metadata residue | gutenberg, ebook, license |
| Chinese pinyin name noise | Cross-book pinyin fragments | chen, yang, zhao |
| Common English personal names | Generic personal names | john, alice, david, mary |

### Section 4: Interrogative Words (~11 words)

`who, what, why, where, when, which, whose, whom, how, whether, if`

### Section 5: Common English Stopwords

Standard NLP stopwords, covering: pronouns, determiners, prepositions, conjunctions, auxiliary / modal verbs, contraction residue, common adverbs, and full inflectional variations of high-frequency narrative verbs.

### Section 6: Character-Name Noise Discovered in Topic-Modeling Visualization (~354 words)

Character names identified and added by manually inspecting topic keywords in actual BERTopic runs. These words appeared in topic keywords after the initial modeling pass and, upon item-by-item verification as character names, were added to the list.

**Addition rules** (written as comments in the file):
- Confirmed as a personal / character name → add
- Ordinary English word (e.g., bridge) → **do not add**
- Disciplinary term (e.g., orogeny, axon) → **do not add**
- Chinese pinyin lexical item (e.g., dan = 丹) → **do not add**

---

## 3. How It Works

Stopwords are passed to BERTopic via the `stop_words` parameter of `sklearn.feature_extraction.text.CountVectorizer`:

```python
# Key code in topic_modeling.py
from stopwords import CUSTOM_STOPWORDS

stop_words_list = CUSTOM_STOPWORDS.copy()
vectorizer_model = CountVectorizer(stop_words=stop_words_list)
topic_model = BERTopic(vectorizer_model=vectorizer_model, ...)
```

**Note**: Stopword filtering occurs during the **c-TF-IDF computation stage** (i.e., the bag-of-words stage), not during the embedding stage. Therefore, stopwords do not affect document clustering (which is determined by Sentence-BERT embeddings) — they only affect **keyword extraction** for each topic.

---

## 4. Confirmed Retained Important Science Fiction Vocabulary

The following vocabulary has been manually verified as **not** included in the stopwords list, and can participate normally in topic keyword extraction:

| Category | Retained Vocabulary |
|----------|---------------------|
| Book titles | foundation, neuromancer, hyperion, ringworld, solaris, rama |
| Core science fiction concepts | robot, android, cyborg, machine, technology, science, alien, planet, galaxy, universe, consciousness, memory, identity, existence, reality, human, humanity, posthuman, transhuman |
| Social science fiction | dystopia, utopia, rebellion, revolution, war, government, empire |
| Technology terms | quantum, neural, genetic, clone, nano, laser, plasma, virtual, digital, cyberspace, matrix, data, network |

A total of **244 important science fiction terms** are not stopped.

---

## 5. Record of Erroneous Stopword Corrections

Upon item-by-item manual review, **25 erroneously stopped words have been removed** from the stopwords list (see `Final_Erroneous_Stopwords_List.md` for details):

| Category | Words Removed | Reason for Removal |
|----------|---------------|---------------------|
| Book titles / core concepts | dune, ender, murderbot, sophon, sophons, wallfacers, orogene, orogenes | Work titles or core setting terms with topic-discriminating value |
| Transhumanism core words | ai, mind, self, thought, thoughts, thinking, alive, real | Key concepts in transhumanism research |
| Science fiction core words | space, time, energy, power, information, ray | Core science fiction vocabulary |
| Social science fiction words | state, captain, past | Social science fiction / time-concept words |

---

## 6. Maintenance Guidelines

### Adding New Stopwords

1. Add the new word (in lowercase) under the corresponding category section in `stopwords.py`
2. Attach a comment when adding it, explaining the source (which book, which character)
3. **Before adding, verify**:
   - Is the word an ordinary English word? → do not add
   - Is the word a disciplinary term? → do not add
   - Is the word meaningful for topic analysis? → do not add
4. Re-run `topic_modeling.py` for the changes to take effect

### Removing Stopwords

When you discover that a stopword is actually valuable for topic clustering:
1. Delete the word from `stopwords.py`
2. Add a comment at the corresponding location explaining the reason for removal
3. Record the change in the correction document

### Verification Methods

```bash
# Check total number of stopwords
python3 -c "
from bertopic_analysis.stopwords import CUSTOM_STOPWORDS
print(f'Total entries: {len(CUSTOM_STOPWORDS)}')
print(f'Unique words: {len(set(CUSTOM_STOPWORDS))}')
"

# Check whether a given word is in the stopwords list
python3 -c "
from bertopic_analysis.stopwords import CUSTOM_STOPWORDS
word = 'consciousness'  # Replace with the word to check
print(f'{word} is a stopword: {word in CUSTOM_STOPWORDS}')
"
```
