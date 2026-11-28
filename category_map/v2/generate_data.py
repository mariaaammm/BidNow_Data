import argparse
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


CONDITIONS = [
    "new", "used", "like new", "almost new", "brand new",
    "good condition", "excellent condition", "barely used", "original",
    "nor", "ogtagorcvac", "lav vijakum", "gerazanc vijakum",
    "idealakan vijakum", "qich ogtagorcvac"
]

TAILS = [
    "urgent", "negotiable", "price negotiable", "with warranty",
    "delivery available", "free delivery", "best price", "must sell", "cheap",
    "shtap", "sakarkeli", "giny sakarkeli", "erashxiqov",
    "araqelutyamb", "anvchar araqum", "lav gin", "shtap vacharvum e", "ezhan"
]

CITIES = [
    "Yerevan", "Gyumri", "Vanadzor", "Abovyan", "Hrazdan",
    "Vagharshapat", "Ejmiatsin", "Ashtarak", "Kapan", "Armavir",
    "Kentron", "Arabkir", "Nor Nork", "Ajapnyak", "Malatia",
    "Erebuni", "Davtashen", "Qanaqer Zeytun", "Avan", "Shengavit"
]

COLORS = [
    "black", "white", "red", "blue", "green",
    "grey", "gray", "silver", "gold", "brown", "pink", "beige",
    "sev", "spitak", "karmir", "kapuyt", "kanach",
    "moxraguyn", "arcataguyn", "voskeguyn", "shaganakaguyn",
    "vardaguyn", "bezh"
]

SKIP_WORDS = {
    "other", "everything else", "accessories", "parts",
    "parts and accessories", "accessories and parts", "supplies",
    "other items", "miscellaneous", "unisex", "for men", "for women",
    "ayl", "ayl baner", "amen inch", "aqsesuarner",
    "pahestamaser", "tarber", "kananc hamar", "tghamardkanc hamar"
}


def extras_for(main, rng):
    if main == "Real Estate":
        return rng.choice([
            f"{rng.randint(1, 5)} room", f"{rng.randint(1, 5)} senyak",
            f"{rng.randint(1, 4)} bedroom", f"{rng.randint(25, 250)} sq m",
            f"{rng.randint(25, 250)} qm", f"{rng.randint(1, 16)} floor",
            f"{rng.randint(1, 16)} hark", "furnished", "kahavorvac",
            "renovated", "veranorogvac", "kapital veranorogvac",
            "city center", "new building", "norakaruyc", "balcony", "patshgamb"
        ])
    if main == "Vehicles":
        return rng.choice([
            str(rng.randint(1995, 2025)), f"{rng.randint(10, 300)}k km",
            f"{rng.randint(10, 300)} hazar km", "automatic", "avtomat",
            "manual", "mexanikakan", "hybrid", "electric", "petrol", "benzin",
            "diesel", "gaz", "4x4", "full option", rng.choice(COLORS)
        ])
    if main == "Electronics":
        return rng.choice([
            f"{rng.choice([32, 64, 128, 256, 512])}GB", "1TB",
            f"{rng.choice([8, 16, 32])}GB RAM", "with box", "tupov",
            "with charger", "licqavorichov", "unlocked", "sim free",
            "battery 100%", "no scratches", "aranc xaz", rng.choice(COLORS)
        ])
    if main == "Appliances":
        return rng.choice([
            "with warranty", "erashxiqov", "A+ class", "A++",
            "inverter", "no frost", "built in", "nerkarucvox",
            f"{rng.randint(1, 5)} years old", "low energy", rng.choice(COLORS)
        ])
    if main in ("Clothing and Fashion", "Baby and Kids"):
        return rng.choice([
            f"size {rng.choice(['XS', 'S', 'M', 'L', 'XL', 'XXL'])}",
            f"size {rng.randint(36, 46)}", f"chaps {rng.randint(36, 46)}",
            "cotton", "bambak", "leather", "kashve", "winter", "dzmerayin",
            "summer", "amarayin", "for kids", "mankakan", rng.choice(COLORS)
        ])
    if main == "Home and Garden":
        return rng.choice([
            "wooden", "payte", "metal", "modern", "classic",
            "foldable", "calvox", "handmade", "dzeragorc",
            f"{rng.randint(50, 220)} cm", "set of 4", "komplekt", rng.choice(COLORS)
        ])
    if main == "Renovation and Construction":
        return rng.choice([
            "wholesale", "mecacax", "retail", "manracax",
            "waterproof", "jradimackun", "heavy duty", "imported",
            f"{rng.randint(1, 50)} kg", f"{rng.randint(1, 20)} m", "new in pack"
        ])
    if main == "Health and Beauty":
        return rng.choice([
            "organic", "natural", "bnakan", "hypoallergenic",
            f"{rng.choice([30, 50, 100, 150, 200])} ml", "sealed",
            "pak tupov", "original brand", "for dry skin", "for all skin types"
        ])
    if main == "Hobbies and Sports":
        return rng.choice([
            "professional", "beginner", "sksnakneri hamar",
            "lightweight", "tetev", "carbon", "aluminum",
            f"{rng.choice([24, 26, 27, 29])} inch", "outdoor", rng.choice(COLORS)
        ])
    if main == "Food and Beverages":
        return rng.choice([
            "homemade", "tnakan", "organic", "fresh", "tarm",
            "natural", "bnakan", "from village", "gyughakan",
            f"{rng.randint(1, 25)} kg", f"{rng.randint(1, 10)} L", "wholesale"
        ])
    if main == "Pets and Animals":
        return rng.choice([
            "pure breed", "maqracegh", "vaccinated", "patvastvac",
            "young", "dekorativ", f"{rng.randint(2, 24)} months",
            f"{rng.randint(2, 24)} amsekan", "with passport", "pasportov"
        ])
    if main == "Jobs":
        return rng.choice([
            "full time", "amboghj druyq", "part time", "kes druyq",
            "remote", "heravar", "flexible schedule", "azat grafik",
            f"salary {rng.randint(1, 9) * 100000} AMD",
            f"ashxatavardz {rng.randint(150, 800)} hazar",
            "no experience", "aranc pordzi", "experience required"
        ])
    if main == "Services":
        return rng.choice([
            "professional", "experienced", "pordzaru", "affordable",
            "matcheli", "all Yerevan", "amboghj Yerevanum",
            "same day", "arag", "free visit", "anvchar ayc", "24/7"
        ])
    if main in ("Collectibles", "Business and Equipment"):
        return rng.choice([
            "vintage", "antique", "hnadaryan", "rare", "hazvagyut",
            "industrial", "artadrakan", "commercial", "working condition",
            "ashxatogh vijakum", f"{rng.randint(1950, 2023)}"
        ])
    return rng.choice([
        rng.choice(COLORS), "handmade", "dzeragorc", "imported",
        "high quality", "vorakyal", f"{rng.randint(2, 12)} pcs", "set", "komplekt"
    ])


def add_typo(text, rng):
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    k = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap":
        w = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    elif op == "drop":
        w = w[:k] + w[k + 1:]
    else:
        w = w[:k] + w[k] + w[k:]
    words[i] = w
    return " ".join(words)


def make_title(seed_word, main_cat, rng):
    parts = [seed_word]
    if rng.random() < 0.35:
        parts.insert(0, rng.choice(CONDITIONS))
    for _ in range(rng.randint(0, 2)):
        parts.append(extras_for(main_cat, rng))
    if rng.random() < 0.25:
        parts.append(rng.choice(TAILS))
    if rng.random() < 0.30:
        parts.append(rng.choice(CITIES))

    title = " ".join(parts)
    if rng.random() < 0.12:
        title = add_typo(title, rng)
    return title.lower() if rng.random() < 0.5 else title.capitalize()


def get_seeds_by_category(categories, keywords):
    seeds_map = {}
    for main, subs in categories.items():
        for sub, leaves in subs.items():
            path = f"{main} > {sub}"
            words = set(k.lower() for k in keywords.get(path, []))
            for leaf in leaves:
                if leaf.lower() not in SKIP_WORDS:
                    words.add(leaf.lower())
            words = [w for w in words if w and w not in SKIP_WORDS]
            if words:
                seeds_map[path] = sorted(words)
    return seeds_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dir", default=str(PARENT))
    parser.add_argument("--per-sub", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(PARENT / "data" / "category_data.csv"))
    args = parser.parse_args()

    sys.path.insert(0, args.map_dir)
    from category_preprocessing import CATEGORIES, KEYWORDS

    rng = random.Random(args.seed)
    seeds_map = get_seeds_by_category(CATEGORIES, KEYWORDS)

    rows = []
    for path, words in seeds_map.items():
        main_cat = path.split(" > ", 1)[0]
        if len(words) >= 4:
            shuffled = words[:]
            rng.shuffle(shuffled)
            cut = max(1, int(len(shuffled) * 0.2))
            pools = {"test": shuffled[:cut], "train": shuffled[cut:]}
            per_split = {"train": int(args.per_sub * 0.8), "test": int(args.per_sub * 0.2)}
            for split_name, pool in pools.items():
                for _ in range(per_split[split_name]):
                    w = rng.choice(pool)
                    rows.append({"title": make_title(w, main_cat, rng), "category": path, "split": split_name})
        else:
            for i in range(args.per_sub):
                w = rng.choice(words)
                split_name = "test" if i < int(args.per_sub * 0.2) else "train"
                rows.append({"title": make_title(w, main_cat, rng), "category": path, "split": split_name})

    df = pd.DataFrame(rows).drop_duplicates(subset=["title"]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved {len(df)} rows across {df['category'].nunique()} categories -> {out_path}")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()