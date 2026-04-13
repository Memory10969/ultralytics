
# htmb hierarchical text memory bank

"""
TextSpace v3: 6-level text hierarchy with upward-only aggregation.

Levels (coarse → fine):
    domain → category → subcategory → entity → phrase → description

Each tag is auto-assigned a level:
    - category:    tag in CATEGORY_ANCHORS and maps to itself (e.g. "animal"→"animal")
    - subcategory: tag in CATEGORY_ANCHORS but maps to other  (e.g. "mammal"→"animal")
    - entity:      everything else                             (e.g. "dog")

Usage:
    ts = TextSpace("ram_tag_list.txt")
    ts.save_tree("tree_cache.json")

    # Instant load next time
    ts = TextSpace.from_tree("ram_tag_list.txt", "tree_cache.json")

    result = ts.query("a big golden retriever playing in park")
"""

import json
import re
from collections import defaultdict, Counter
from typing import Optional, Dict


# ══════════════════════════════════════════════════════════════
# SubjectExtractor
# ══════════════════════════════════════════════════════════════

import os 
current_dir = os.path.dirname(os.path.abspath(__file__))



class SubjectExtractor:
    """Extract main subject noun from text using spaCy."""

    def __init__(self, nlp=None):
        if nlp is not None:
            self._nlp = nlp
        else:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm",
                                       disable=["ner", "textcat", "lemmatizer"])
            except Exception:
                self._nlp = None

    @property
    def available(self) -> bool:
        return self._nlp is not None

    def extract_first_noun(self, text: str) -> str:
        if self._nlp is None:
            return text
        doc = self._nlp(text)
        chunks = list(doc.noun_chunks)
        if chunks:
            return self._head_with_compound(chunks[0].root)
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                return self._head_with_compound(token)
        return doc.text

    def extract_first_chunk(self, text: str) -> str:
        if self._nlp is None:
            return text
        doc = self._nlp(text)
        chunks = list(doc.noun_chunks)
        return chunks[0].text if chunks else doc.text

    @staticmethod
    def _head_with_compound(token) -> str:
        parts = []
        for child in token.children:
            if child.dep_ in ("compound", "amod") and child.i < token.i:
                parts.append(child.text)
        parts.append(token.text)
        return " ".join(parts)


# ══════════════════════════════════════════════════════════════
# TextSpace
# ══════════════════════════════════════════════════════════════

class TextSpace:

    MAX_HYPERNYM_DEPTH = 12

    # ── Domain ← Category mapping ──
    DOMAIN_MAP = {
        "person": "living_thing",
        "animal": "living_thing",
        "plant": "living_thing",
        "food": "living_thing",
        "vehicle": "object",
        "furniture": "object",
        "clothing": "object",
        "tool": "object",
        "container": "object",
        "electronics": "object",
        "kitchenware": "object",
        "decoration": "object",
        "building": "place",
        "nature": "place",
        "sports": "abstract",
        "other": "abstract",
    }

    # ── Category anchors in WordNet hypernym chains ──
    CATEGORY_ANCHORS = {
        # --- person ---
        "person": "person", "people": "person", "human": "person",
        "someone": "person", "somebody": "person",
        "worker": "person", "skilled worker": "person",
        "performer": "person", "entertainer": "person",
        "creator": "person", "artist": "person",
        "communicator": "person", "leader": "person",
        "contestant": "person", "disputant": "person",
        "intellectual": "person", "professional": "person",
        "adult": "person", "juvenile": "person", "child": "person",
        "male": "person", "female": "person",
        "traveler": "person", "relative": "person",
        # --- animal ---
        "animal": "animal", "vertebrate": "animal", "invertebrate": "animal",
        "mammal": "animal", "bird": "animal", "fish": "animal",
        "reptile": "animal", "amphibian": "animal", "insect": "animal",
        "arthropod": "animal", "mollusk": "animal", "chordate": "animal",
        "domestic animal": "animal", "canine": "animal", "feline": "animal",
        "primate": "animal", "rodent": "animal", "ungulate": "animal",
        "carnivore": "animal", "herbivore": "animal",
        "aquatic vertebrate": "animal", "worm": "animal",
        "crustacean": "animal", "arachnid": "animal",
        # --- food ---
        "food": "food", "nutrient": "food", "dish": "food",
        "fruit": "food", "vegetable": "food", "beverage": "food",
        "drink": "food", "alcohol": "food", "foodstuff": "food",
        "produce": "food", "baked goods": "food", "meat": "food",
        "seafood": "food", "dairy product": "food", "dessert": "food",
        "condiment": "food", "flavorer": "food", "sweetening": "food",
        # --- plant ---
        "plant": "plant", "flora": "plant",
        "tree": "plant", "flower": "plant", "herb": "plant",
        "shrub": "plant", "vascular plant": "plant", "woody plant": "plant",
        "moss": "plant", "fern": "plant", "grass": "plant",
        # --- vehicle ---
        "vehicle": "vehicle", "conveyance": "vehicle", "craft": "vehicle",
        "wheeled vehicle": "vehicle", "motor vehicle": "vehicle",
        "vessel": "vehicle", "aircraft": "vehicle", "boat": "vehicle",
        "self-propelled vehicle": "vehicle",
        # --- furniture ---
        "furniture": "furniture", "furnishing": "furniture",
        "bedroom furniture": "furniture", "office furniture": "furniture",
        "seat": "furniture", "table": "furniture",
        # --- clothing ---
        "clothing": "clothing", "garment": "clothing", "attire": "clothing",
        "wear": "clothing", "footwear": "clothing",
        "headdress": "clothing", "nightwear": "clothing",
        "outerwear": "clothing", "overgarment": "clothing",
        # --- tool ---
        "tool": "tool", "implement": "tool", "utensil": "tool",
        "hand tool": "tool", "cutting implement": "tool",
        "measuring instrument": "tool", "optical instrument": "tool",
        "weapon": "tool", "arm": "tool",
        # --- container ---
        "container": "container", "receptacle": "container",
        "box": "container", "bag": "container", "case": "container",
        # --- building ---
        "building": "building", "establishment": "building",
        "place of business": "building", "institution": "building",
        "dwelling": "building", "house": "building",
        "mercantile establishment": "building",
        "room": "building", "enclosure": "building",
        # --- nature ---
        "geological formation": "nature", "body of water": "nature",
        "natural object": "nature", "natural elevation": "nature",
        "shore": "nature", "natural depression": "nature",
        "tract": "nature", "parkland": "nature",
        "geographical area": "nature", "region": "nature",
        # --- electronics ---
        "electronic equipment": "electronics",
        "electronic device": "electronics",
        "computer": "electronics", "telephone": "electronics",
        "display": "electronics", "monitor": "electronics",
        # --- sports ---
        "sport": "sports", "athletic game": "sports",
        "field game": "sports", "contact sport": "sports",
        "outdoor game": "sports", "outdoor sport": "sports",
        # --- decoration ---
        "decoration": "decoration", "ornament": "decoration",
        "artwork": "decoration", "picture": "decoration",
        "representation": "decoration", "sculpture": "decoration",
        # --- musical instrument → tool ---
        "musical instrument": "tool", "stringed instrument": "tool",
        "wind instrument": "tool", "percussion instrument": "tool",
        "keyboard instrument": "tool",
    }

    # ── Synset override: pick specific WordNet sense ──
    SYNSET_OVERRIDES = {
        "park": ("park.n.01", "building"),
        "couch": ("couch.n.01", "furniture"),
        "bat": ("bat.n.05", "tool"),
        "mouse": ("mouse.n.01", "animal"),
        "monitor": ("monitor.n.04", "electronics"),
        "speaker": ("speaker.n.02", "electronics"),
        "crane": ("crane.n.02", "animal"),
        "bass": ("bass.n.07", "animal"),
        "mole": ("mole.n.06", "animal"),
        "seal": ("seal.n.09", "animal"),
        "perch": ("perch.n.03", "animal"),
        "kite": ("kite.n.03", "tool"),
        "ram": ("ram.n.05", "animal"),
        "star": ("star.n.01", "nature"),
        "pipe organ": ("organ.n.01", "tool"),
        "palm tree": ("palm.n.01", "plant"),
        "spring": ("spring.n.01", "nature"),
        "bowl": ("bowl.n.01", "kitchenware"),
        "pitch": ("pitch.n.01", "sports"),
        "tank": ("tank.n.01", "container"),
        "board": ("board.n.09", "tool"),
        "chest": ("chest.n.01", "other"),
        "iris": ("iris.n.01", "plant"),
    }

    # ── Category override: force category regardless of WordNet chain ──
    # More robust than SYNSET_OVERRIDES — always works
    CATEGORY_OVERRIDES = {
        # Physical-object sense, not figurative/person sense
        "light": "electronics",
        "street light": "electronics",
        "fence": "nature",
        "window": "building",
        "drawer": "furniture",
        "plate": "kitchenware",
        "hose": "tool",
        "brick": "other",
        "pillar": "building",
        "dresser": "furniture",
        "trash": "other",
        "cup": "kitchenware",
        "wave": "nature",
        "mug": "kitchenware",
        "pot": "kitchenware",
        "bumper": "other",
        "top": "other",
        "sticker": "other",
        "letter": "other",
        # Colors — not person/radical/scientist
        "red": "other",
        "gray": "other",
        "grey": "other",
        "white": "other",
        "black": "other",
        "brown": "other",
        "orange": "other",
        "pink": "other",
        "purple": "other",
        "silver": "other",
        "golden": "other",
        # Common physical objects mis-categorized by WordNet
        "panel": "other",
        "curtain": "other",
        "carpet": "nature",
        "wire": "other",
        "rope": "tool",
        "chain": "tool",
        "mat": "other",
        "pad": "other",
        "button": "other",
    }

    # ── Irregular plurals ──
    IRREGULAR_PLURALS = {
        "people": "person", "men": "man", "women": "woman",
        "children": "child", "mice": "mouse", "geese": "goose",
        "feet": "foot", "teeth": "tooth", "oxen": "ox",
        "leaves": "leaf", "knives": "knife", "wolves": "wolf",
        "lives": "life", "wives": "wife", "halves": "half",
        "calves": "calf", "loaves": "loaf", "shelves": "shelf",
        "potatoes": "potato", "tomatoes": "tomato",
        "heroes": "hero", "cacti": "cactus",
        "fungi": "fungus", "octopi": "octopus",
        "phenomena": "phenomenon", "criteria": "criterion",
        "bases": "base", "crises": "crisis", "oases": "oasis",
    }

    # ── Regex for stripping modifiers ──
    _RE_DETERMINERS = re.compile(
        r"^(a |an |the |this |that |these |those |my |his |her |our |their "
        r"|its |some |any )+",
        re.IGNORECASE,
    )
    _RE_QUANTIFIERS = re.compile(
        r"^(one |two |three |four |five |six |seven |eight |nine |ten |"
        r"eleven |twelve |thirteen |twenty |thirty |hundred |thousand |"
        r"several |many |few |couple |couple of |pair of |group of |bunch of |"
        r"lot of |lots of |number of |"
        r"first |second |third |last |next |other |another )+",
        re.IGNORECASE,
    )
    _RE_ADJECTIVES = re.compile(
        r"^(big |small |large |little |tiny |huge |tall |short |long |"
        r"old |young |new |"
        r"red |blue |green |black |white |yellow |brown |pink |gray |grey |"
        r"orange |purple |"
        r"bright |dark |light |deep |pale |golden |silver |"
        r"beautiful |pretty |ugly |nice |good |bad |great |amazing |"
        r"hot |cold |warm |cool |wet |dry |"
        r"left |right |top |bottom |front |back |middle |central |rear |"
        r"whole |entire |full |empty |single |double |thin |thick |heavy |"
        r"round |flat |"
        r"wooden |metal |glass |plastic |stone |concrete |"
        r"modern |classic |ancient |vintage |rustic )+",
        re.IGNORECASE,
    )

    # ══════════════════════════════════════════════════════════
    # Init
    # ══════════════════════════════════════════════════════════

    def __init__(self, tag_list_path: str=os.path.join(current_dir, "tag_list.txt"), log: bool = True):
        self.log = log
        self._tag_list_path = tag_list_path

        with open(tag_list_path) as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.tag_set = set(t.lower() for t in self.tags)
        self._tag_lookup = {t.lower(): t for t in self.tags}
        self._log(f"Loaded {len(self.tags)} tags")

        self._tree: Dict[str, dict] = {}
        self._category_members = defaultdict(list)
        self._build_tree()

        self._norm_cache: Dict[str, str] = {}
        self._build_norm_cache()

        self._known_domains = set(self.DOMAIN_MAP.values())

        self._subj = SubjectExtractor()
        if self._subj.available:
            self._log("SubjectExtractor ready (spaCy)")

        self._norm_failures = Counter()
        self._norm_failure_ctx = {}

        self._log("TextSpace ready")

    # ══════════════════════════════════════════════════════════
    # Build Tree
    # ══════════════════════════════════════════════════════════

    def _build_tree(self):
        from nltk.corpus import wordnet as wn
        self._wn = wn
        self._log("Building WordNet tree...")

        for tag in self.tags:
            tag_lower = tag.lower()

            # 1. Determine synset & category
            synset, forced_cat = self._find_best_synset(tag)

            # 2. CATEGORY_OVERRIDES has highest priority
            if tag_lower in self.CATEGORY_OVERRIDES:
                forced_cat = self.CATEGORY_OVERRIDES[tag_lower]

            if synset is None:
                category = forced_cat or "other"
                domain = self.DOMAIN_MAP.get(category, "abstract")
                tag_level = self._detect_tag_level(tag_lower, category)
                self._tree[tag] = {
                    "tag_level": tag_level,
                    "category": category,
                    "subcategory_chain": [],
                    "domain": domain,
                    "source": "no_synset",
                }
            else:
                chain = self._walk_hypernyms(synset)
                category = forced_cat or self._detect_category(chain)
                domain = self.DOMAIN_MAP.get(category, "abstract")
                subcat_chain = self._extract_subcategory_chain(chain, category)
                tag_level = self._detect_tag_level(tag_lower, category)

                self._tree[tag] = {
                    "tag_level": tag_level,
                    "category": category,
                    "subcategory_chain": subcat_chain,
                    "domain": domain,
                    "source": "wordnet",
                }

            self._category_members[category].append(tag)

        # Log stats
        level_counts = defaultdict(int)
        cat_counts = defaultdict(int)
        for node in self._tree.values():
            level_counts[node["tag_level"]] += 1
            cat_counts[node["category"]] += 1
        self._log(f"Tree built: {len(self._tree)} tags")
        self._log(f"  tag_levels: {dict(level_counts)}")
        self._log(f"  categories: {dict(cat_counts)}")

    def _detect_tag_level(self, tag_lower: str, category: str) -> str:
        """
        Auto-detect what level a tag belongs to:
          - category:    in CATEGORY_ANCHORS and maps to itself
          - subcategory: in CATEGORY_ANCHORS but maps to different value
          - entity:      everything else
        """
        if tag_lower in self.CATEGORY_ANCHORS:
            if self.CATEGORY_ANCHORS[tag_lower] == tag_lower:
                return "category"
            else:
                return "subcategory"
        return "entity"

    def _extract_subcategory_chain(self, chain: list, category: str) -> list:
        """Intermediate nodes between entity (chain[0]) and category anchor."""
        anchor_idx = None
        for i, node in enumerate(chain):
            if self.CATEGORY_ANCHORS.get(node["lemma"]) == category:
                anchor_idx = i
                break
        if anchor_idx is None or anchor_idx <= 1:
            return []
        return [node["lemma"] for node in chain[1:anchor_idx]]

    def _find_best_synset(self, tag):
        wn = self._wn
        tag_lower = tag.lower()

        if tag_lower in self.SYNSET_OVERRIDES:
            synset_name, forced_cat = self.SYNSET_OVERRIDES[tag_lower]
            try:
                return wn.synset(synset_name), forced_cat
            except Exception:
                pass

        key = tag_lower.replace(" ", "_")
        synsets = wn.synsets(key, pos=wn.NOUN)
        if not synsets:
            for word in reversed(tag_lower.split()):
                synsets = wn.synsets(word, pos=wn.NOUN)
                if synsets:
                    break
        if not synsets:
            return None, None
        if len(synsets) == 1:
            return synsets[0], None

        best_syn, best_score = synsets[0], -1
        for idx, syn in enumerate(synsets[:8]):
            chain = self._walk_hypernyms(syn)
            score = 0
            for i, node in enumerate(chain):
                if node["lemma"] in self.CATEGORY_ANCHORS:
                    score = 200 - i * 10
                    break
            if syn.lemmas()[0].name().replace("_", " ").lower() == tag_lower:
                score += 50
            for word in tag_lower.split():
                if word in syn.definition().lower():
                    score += 5
            score -= idx * 3
            if score > best_score:
                best_score = score
                best_syn = syn
        return best_syn, None

    def _walk_hypernyms(self, synset, max_depth=None):
        max_depth = max_depth or self.MAX_HYPERNYM_DEPTH
        chain, current, visited = [], synset, set()
        while current and len(chain) < max_depth:
            if current.name() in visited:
                break
            visited.add(current.name())
            lemma = current.lemmas()[0].name().replace("_", " ").lower()
            chain.append({"lemma": lemma, "synset": current.name()})
            hypernyms = current.hypernyms()
            current = hypernyms[0] if hypernyms else None
        return chain

    def _detect_category(self, chain):
        for node in chain:
            if node["lemma"] in self.CATEGORY_ANCHORS:
                return self.CATEGORY_ANCHORS[node["lemma"]]
        return "other"

    # ══════════════════════════════════════════════════════════
    # Build Normalization Cache
    # ══════════════════════════════════════════════════════════

    def _build_norm_cache(self):
        """
        Build text → tag lookup.
        Only exact tags + plural/singular variants.
        NO tail-word registration (avoids "ground"→"training ground" etc.)
        """
        self._log("Building normalization cache...")
        variant_count = 0

        for tag in self.tags:
            tl = tag.lower()
            self._norm_cache[tl] = tag

            # Plural/singular variants of the full tag
            words = tl.split()
            last = words[-1]
            prefix = " ".join(words[:-1]) if len(words) > 1 else ""

            for variant in self._to_plurals(last) + self._to_singulars(last):
                key = f"{prefix} {variant}".strip() if prefix else variant
                if key not in self._norm_cache:
                    self._norm_cache[key] = tag
                    variant_count += 1

        self._log(f"  {len(self.tags)} tags + {variant_count} variants")

    def _to_plurals(self, word: str) -> list:
        plurals = []
        for plural, singular in self.IRREGULAR_PLURALS.items():
            if singular == word:
                plurals.append(plural)
        if word.endswith(("s", "sh", "ch", "x", "z")):
            plurals.append(word + "es")
        elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            plurals.append(word[:-1] + "ies")
        elif word.endswith("f"):
            plurals.append(word[:-1] + "ves")
        elif word.endswith("fe"):
            plurals.append(word[:-2] + "ves")
        else:
            plurals.append(word + "s")
        return plurals

    def _to_singulars(self, word: str) -> list:
        singulars = []
        if word in self.IRREGULAR_PLURALS:
            singulars.append(self.IRREGULAR_PLURALS[word])
        if word.endswith("ies") and len(word) > 3:
            singulars.append(word[:-3] + "y")
        if word.endswith("ves"):
            singulars.append(word[:-3] + "f")
            singulars.append(word[:-3] + "fe")
        if word.endswith("es") and len(word) > 2:
            singulars.append(word[:-2])
        if word.endswith("s") and not word.endswith("ss"):
            singulars.append(word[:-1])
        return singulars

    # ══════════════════════════════════════════════════════════
    # Query
    # ══════════════════════════════════════════════════════════

    def query(self, text: str) -> dict:
        """
        Map input text to 6-level hierarchy. Only upward aggregation.
        All values are lists except input_level (str) and source (str).

        source: "tag_list" | "wordnet" | "none"
        """
        text = text.strip()
        tl = text.lower()

        # 1. Direct tag match → use tag's own level
        if tl in self._norm_cache:
            tag = self._norm_cache[tl]
            tag_level = self._tree.get(tag, {}).get("tag_level", "entity")
            return self._tag_result(tag, input_level=tag_level)

        # 2. Domain match
        if tl in self._known_domains:
            return self._result(domain=[tl], input_level="domain",
                                source="tag_list")

        # 3. Description level: try tag_list match
        tag, phrase, raw_noun = self._extract_match(text)

        if tag is not None:
            r = self._tag_result(tag, input_level="description")
            if phrase:
                r["phrase"] = [phrase]
            r["description"] = [text]
            return r

        # 4. WordNet fallback: resolve raw_noun via WordNet directly
        if raw_noun:
            wn_info = self._wordnet_resolve(raw_noun)
            if wn_info:
                r = self._result(
                    domain=[wn_info["domain"]],
                    category=[wn_info["category"]],
                    subcategory=wn_info.get("subcategory_chain", []),
                    entity=[raw_noun],
                    phrase=[phrase] if phrase else [],
                    description=[text],
                    input_level="description",
                    source="wordnet",
                )
                return r

        # Nothing matched — record failure
        failed = raw_noun or tl
        if failed and len(failed) > 1:
            self._norm_failures[failed] += 1
            if failed not in self._norm_failure_ctx:
                self._norm_failure_ctx[failed] = text
            if self._norm_failures[failed] == 1:
                self._log(f"⚠ RESOLVE FAIL: \"{failed}\" (from \"{text}\")")
        return self._result(description=[text], input_level="description")

    def _tag_result(self, tag: str, input_level: str) -> dict:
        """Build result from resolved tag, filling upward only."""
        node = self._tree.get(tag, {})
        tag_level = node.get("tag_level", "entity")
        category = node.get("category", "other")
        domain = node.get("domain", self.DOMAIN_MAP.get(category, "abstract"))
        subcat_chain = node.get("subcategory_chain", [])

        r = self._result(input_level=input_level, source="tag_list")
        r["domain"] = [domain]
        r["category"] = [category]

        if tag_level == "category":
            pass  # only domain + category filled
        elif tag_level == "subcategory":
            r["subcategory"] = [tag.lower()]
        else:  # entity
            r["subcategory"] = list(subcat_chain)
            r["entity"] = [tag]

        return r

    def _extract_match(self, text: str):
        """
        From description text, try to match a tag from tag_list.
        Returns (tag, phrase, raw_noun):
          - tag found:    (tag, phrase, None)
          - tag not found: (None, phrase, raw_noun_for_wn_fallback)
        """
        phrase = None
        raw_noun = None

        if self._subj.available:
            noun = self._subj.extract_first_noun(text)
            phrase = self._subj.extract_first_chunk(text)
            tag = self._normalize(noun, full_text=text)
            if tag:
                return tag, phrase, None
            # Keep cleaned noun for WordNet fallback
            raw_noun = self._strip_modifiers(noun)
        else:
            phrase = text

        # Fallback: try _normalize on full text
        tag = self._normalize(text, full_text=text)
        if tag:
            return tag, phrase or text, None

        # No tag_list match — prepare raw_noun for WordNet fallback
        if raw_noun is None:
            raw_noun = self._strip_modifiers(text)

        return None, phrase, raw_noun

    def _strip_modifiers(self, text: str) -> str:
        """Strip determiners, quantifiers, adjectives. Return cleaned noun."""
        s = text.lower().strip()
        for pattern in [self._RE_DETERMINERS, self._RE_QUANTIFIERS,
                        self._RE_ADJECTIVES]:
            s = pattern.sub("", s).strip()
        # Try to singularize the last word
        words = s.split()
        if words:
            for singular in self._to_singulars(words[-1]):
                if singular != words[-1]:
                    words[-1] = singular
                    break  # take first singular form
            s = " ".join(words)
        return s

    def _wordnet_resolve(self, word: str) -> Optional[dict]:
        """
        Fallback: resolve a word via WordNet when not in tag_list.
        Returns {"category", "subcategory_chain", "domain"} or None.
        """
        # Lazy import WordNet if not loaded (e.g. from_tree mode)
        if self._wn is None:
            try:
                from nltk.corpus import wordnet as wn
                self._wn = wn
            except Exception:
                return None

        wn = self._wn
        word_lower = word.lower().strip()

        # Check CATEGORY_OVERRIDES first
        if word_lower in self.CATEGORY_OVERRIDES:
            cat = self.CATEGORY_OVERRIDES[word_lower]
            return {
                "category": cat,
                "subcategory_chain": [],
                "domain": self.DOMAIN_MAP.get(cat, "abstract"),
            }

        # Try WordNet lookup
        key = word_lower.replace(" ", "_")
        synsets = wn.synsets(key, pos=wn.NOUN)
        if not synsets:
            # Try individual words (right to left)
            for w in reversed(word_lower.split()):
                synsets = wn.synsets(w, pos=wn.NOUN)
                if synsets:
                    break
        if not synsets:
            return None

        # Pick best synset (reuse scoring logic)
        best_syn = synsets[0]
        if len(synsets) > 1:
            best_score = -1
            for idx, syn in enumerate(synsets[:8]):
                chain = self._walk_hypernyms(syn)
                score = 0
                for i, node in enumerate(chain):
                    if node["lemma"] in self.CATEGORY_ANCHORS:
                        score = 200 - i * 10
                        break
                if syn.lemmas()[0].name().replace("_", " ").lower() == word_lower:
                    score += 50
                score -= idx * 3
                if score > best_score:
                    best_score = score
                    best_syn = syn

        chain = self._walk_hypernyms(best_syn)
        category = self._detect_category(chain)
        domain = self.DOMAIN_MAP.get(category, "abstract")
        subcat_chain = self._extract_subcategory_chain(chain, category)

        return {
            "category": category,
            "subcategory_chain": subcat_chain,
            "domain": domain,
        }

    def _normalize(self, chunk: str, full_text: str = "") -> Optional[str]:
        """Try to map a text chunk to a known tag. No side effects."""
        tl = chunk.lower().strip()

        if tl in self._norm_cache:
            return self._norm_cache[tl]

        # Strip modifiers progressively
        stripped = tl
        for pattern in [self._RE_DETERMINERS, self._RE_QUANTIFIERS,
                        self._RE_ADJECTIVES]:
            new = pattern.sub("", stripped).strip()
            if new != stripped:
                stripped = new
                if stripped in self._norm_cache:
                    return self._norm_cache[stripped]

        # Bigrams (right to left), then single words (right to left)
        words = stripped.split()
        for i in range(len(words) - 2, -1, -1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in self._norm_cache:
                return self._norm_cache[bigram]
        for word in reversed(words):
            if word in self._norm_cache:
                return self._norm_cache[word]

        # Retry on original (before stripping)
        orig_words = tl.split()
        if orig_words != words:
            for i in range(len(orig_words) - 2, -1, -1):
                bigram = f"{orig_words[i]} {orig_words[i+1]}"
                if bigram in self._norm_cache:
                    return self._norm_cache[bigram]
            for word in reversed(orig_words):
                if word in self._norm_cache:
                    return self._norm_cache[word]

        return None

    # ══════════════════════════════════════════════════════════
    # Result builder
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _result(domain=None, category=None, subcategory=None,
                entity=None, phrase=None, description=None,
                input_level="description", source="none") -> dict:
        return {
            "domain": domain or [],
            "category": category or [],
            "subcategory": subcategory or [],
            "entity": entity or [],
            "phrase": phrase or [],
            "description": description or [],
            "input_level": input_level,
            "source": source,
        }

    # ══════════════════════════════════════════════════════════
    # Diagnostics
    # ══════════════════════════════════════════════════════════

    def report_missing(self, top_n: int = 50, save_txt: str = None):
        if not self._norm_failures:
            print("[TextSpace] No normalize failures recorded.")
            return
        print(f"\n[TextSpace] NORMALIZE FAILURE REPORT")
        print(f"{'='*70}")
        print(f"Unique: {len(self._norm_failures)}  "
              f"Total: {sum(self._norm_failures.values())}")
        print(f"\nTop {top_n}:")
        print(f"  {'WORD':<30} {'COUNT':>6}  CONTEXT")
        print(f"  {'-'*60}")
        for word, count in self._norm_failures.most_common(top_n):
            ctx = self._norm_failure_ctx.get(word, "")
            print(f"  {word:<30} {count:>6}  \"{ctx}\"")
        if save_txt:
            with open(save_txt, "w") as f:
                for word, _ in self._norm_failures.most_common():
                    f.write(word + "\n")
            print(f"\nWrote {len(self._norm_failures)} words to {save_txt}")

    def report_tag_levels(self):
        """Print all tags grouped by their auto-detected level."""
        by_level = defaultdict(list)
        for tag, node in self._tree.items():
            by_level[node["tag_level"]].append((tag, node["category"]))
        for level in ["category", "subcategory", "entity"]:
            tags = by_level.get(level, [])
            print(f"\n[{level.upper()}] ({len(tags)} tags):")
            for tag, cat in sorted(tags, key=lambda x: (x[1], x[0])):
                print(f"  {cat:<14s}  {tag}")

    # ══════════════════════════════════════════════════════════
    # Convenience
    # ══════════════════════════════════════════════════════════

    def get_node(self, tag: str) -> dict:
        return self._tree.get(tag, {})

    def get_entity(self, text: str, lookup: bool = True) -> Optional[str]:
        """Extract the best matching entity from text.

        Args:
            text:   Input text (tag, phrase, or description).
            lookup: If True and no entity is found, fall back to the
                    finest available up-level (subcategory → category → domain).
                    If False, return None when no entity is found.

        Returns:
            Matched entity string, or a parent-level string when lookup=True,
            or None if nothing could be resolved.
        """
        r = self.query(text)

        if r["entity"]:
            return r["entity"][0]

        if not lookup:
            return None

        # Fall back upward: subcategory → category → domain
        for key in ("subcategory", "category", "domain"):
            if r[key]:
                return r[key][0]
 
        return None

    def get_tags_by_category(self, category: str) -> list:
        return self._category_members.get(category, [])

    def get_all_categories(self) -> list:
        return list(self._category_members.keys())

    def exists(self, tag: str) -> bool:
        return tag.lower() in self.tag_set

    # ══════════════════════════════════════════════════════════
    # Serialize / Deserialize
    # ══════════════════════════════════════════════════════════

    def save_tree(self, path: str):
        with open(path, "w") as f:
            json.dump(self._tree, f, indent=2, ensure_ascii=False)
        self._log(f"Tree saved to {path}")

    @classmethod
    def from_tree(cls, tag_list_path: str, tree_path: str, log: bool = True):
        """Instant load from cached tree. No WordNet needed."""
        obj = object.__new__(cls)
        obj.log = log
        obj._wn = None
        obj._tag_list_path = tag_list_path

        with open(tag_list_path) as f:
            obj.tags = [line.strip() for line in f if line.strip()]
        obj.tag_set = set(t.lower() for t in obj.tags)
        obj._tag_lookup = {t.lower(): t for t in obj.tags}

        with open(tree_path) as f:
            obj._tree = json.load(f)

        obj._category_members = defaultdict(list)
        for tag, node in obj._tree.items():
            obj._category_members[node.get("category", "other")].append(tag)

        obj._norm_cache = {}
        obj._build_norm_cache()

        obj._known_domains = set(cls.DOMAIN_MAP.values())

        obj._subj = SubjectExtractor()
        obj._norm_failures = Counter()
        obj._norm_failure_ctx = {}

        if log:
            print(f"[TextSpace] Loaded from tree: {len(obj._tree)} tags")
        return obj

    def _log(self, msg):
        if self.log:
            print(f"[TextSpace] {msg}")


# ══════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_list", type=str, default=os.path.join(current_dir, "tag_list.txt"))
    parser.add_argument("--save_tree", type=str, default=None)
    parser.add_argument("--load_tree", type=str, default=None)
    parser.add_argument("--report_levels", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    if args.load_tree:
        ts = TextSpace.from_tree(args.tag_list, args.load_tree)
    else:
        ts = TextSpace(args.tag_list)
    print(f"\nInit took {time.time()-t0:.2f}s\n")

    if args.save_tree:
        ts.save_tree(args.save_tree)

    if args.report_levels:
        ts.report_tag_levels()
        print()

    queries = [
        # --- domain ---
        "living_thing",
        "object",
        # --- category (tag_level=category) ---
        "animal",
        "person",
        "building",
        "furniture",
        "food",
        # --- subcategory (tag_level=subcategory) ---
        "mammal",
        "seat",
        "tree",
        # --- entity ---
        "dog",
        "car",
        "traffic light",
        "people",
        "clock",
        # --- FIX: WordNet polysemy ---
        "light",               # was: person/friend
        "red lights",          # was: person/friend
        "fence",               # was: person/trader
        "window",              # was: electronics
        "drawer",              # was: person
        "plate",               # was: decoration
        "cup",                 # was: nature/plant
        "mug",                 # was: vehicle
        "pot",                 # was: vehicle
        "red",                 # was: person/radical
        # --- WordNet fallback (not in tag list) ---
        "desk",                # was: FAIL → now WN → furniture
        "collar",              # was: FAIL → now WN
        "top",                 # was: FAIL → now WN
        "part",                # was: FAIL → now WN
        "coffee maker",        # was: ice maker → now WN
        "steeple",             # was: FAIL → now WN
        "sidewalk",            # was: FAIL → now WN
        "handlebars",          # was: FAIL → now WN
        "rug",                 # was: FAIL → now WN
        "planter",             # was: FAIL → now WN
        # --- normal cases ---
        "a big golden retriever playing in park",
        "man walking a dog",
        "a silver car",
        "two white horses",
        "clock on the building",
        "a wooden bench",
        "fried bacon",
        "french fries",
        "people",
        "people walking on the street with traffic light in background",
        "a man stranding on the chair"
    ]

    for q in queries:
        print("=" * 70)
        print(f"  INPUT: '{q}'")
        r = ts.query(q)
        print(f"  level:  {r['input_level']}    source: {r['source']}")
        for key in ["domain", "category", "subcategory", "entity",
                     "phrase", "description"]:
            val = r[key]
            if val:
                print(f"  {key:<14s}: {val}")
        print()

    ts.report_missing()

    # ── get_entity tests ──
    print("\n" + "=" * 70)
    print("  get_entity() TESTS")
    print("=" * 70)
    entity_tests = [
        # (text, lookup)
        ("dog",                             True),   # direct entity
        ("people",                          True),   # irregular plural → person
        ("mammal",                          True),   # subcategory, no entity → fallback
        ("animal",                          True),   # category tag, no entity → fallback
        ("a big golden retriever in park",  True),   # description → entity
        ("unknown_xyz_word",                True),   # nothing → domain fallback
        ("unknown_xyz_word",                False),  # nothing → None
        ("traffic light",                   True),   # multi-word entity
        ("fried bacon",                     True),   # food entity
        ("desk",                            True),   # not in tag list → WN fallback → category
    ]
    for text, lookup in entity_tests:
        result = ts.get_entity(text, lookup=lookup)
        print(f"  lookup={str(lookup):<5}  '{text}'  →  {result!r}")