"""
IC-4-M0: Synthetic QA data builder.
Generates answerable and unanswerable samples with non-overlapping entities
between train and test splits.
"""

import json
import random
import os
from typing import List, Dict, Tuple


FAKE_COMPANIES = [
    "NexaTech", "QuantumLeap Systems", "Veridian Dynamics", "AtlasCore",
    "BrightForge", "CloudPeak Inc", "DataMosaic", "EcoNova",
    "FlarePath", "GridStone", "Helixion Labs", "IronVista",
    "JasperWind", "KairoSoft", "LumenAxis", "MeridianWorks",
    "NorthBridge AI", "OmniPulse", "Prismatica", "RidgeFlow",
    "SilverArc", "TitanSpark", "UltraNode", "VantageCloud",
    "WaveCrest", "XyloGen", "YellowDome", "ZenithOps",
    "AnchorByte", "BoltStream", "CipherTrail", "DuneLogic",
    "EmberPath", "FrostPeak", "GlimmerBox", "HorizonForge",
    "IvyCore", "JetCircuit", "KaleidoMind", "LunarMesh",
]

FAKE_PEOPLE = [
    "Alice Thornton", "Benjamin Cross", "Catherine Wu", "Daniel Park",
    "Eleanor Cho", "Frank Mueller", "Grace Nakamura", "Henry Zhang",
    "Isabel Fernandez", "James Okafor", "Katherine Lindberg", "Leo Martinez",
    "Maya Patel", "Nathan Rhodes", "Olivia Svensson", "Patrick Kowalski",
    "Quinn Harper", "Rachel Ng", "Samuel O'Brien", "Tessa van der Berg",
    "Uma Krishnan", "Victor Delgado", "Wendy Nakamura", "Xavier Torres",
    "Yuki Tanaka", "Zara Ahmed", "Andre Dubois", "Betty Johansson",
    "Claude Fournier", "Diana Rossi", "Erik Magnusson", "Fatima al-Rashid",
]

FAKE_LOCATIONS = [
    "Port Meridian", "New Halcyon", "Sundell City", "Ashwick",
    "Brightwater Bay", "Coldspring", "Dunmoor", "Eastvale",
    "Fairhaven Point", "Greenhollow", "Highcliff", "Ivymead",
    "Jade Harbor", "Kingsport West", "Lakeside Crossing", "Millbrook",
    "Northcote", "Oakenshade", "Pinecrest", "Redmill",
    "Stonebridge Falls", "Thornbury", "Upperford", "Valemount",
    "Willowdale East", "Yarrow Glen", "Amber Coast", "Bayview Heights",
    "Cedar Ridge", "Deepwood", "Elder Grove", "Foxhall",
]

FAKE_YEARS = list(range(1995, 2026))
FAKE_ATTRIBUTES = [
    "revenue ($M)", "employees", "market_share (%)", "patents",
    "active_users (K)", "growth_rate (%)", "carbon_emissions (tons)",
    "r_and_d_spend ($M)", "customer_satisfaction", "production_volume",
]

TEMPLATES_ANSWERABLE = [
    {
        "template_id": "A1",
        "context_tpl": "{company} is headquartered in {location}. In {year}, the company reported {attr} of {value}.",
        "question_tpl": "What was {company}'s {attr} in {year}?",
        "answer_tpl": "{value}",
        "positive_tpl": "{company} reported {attr} of {value} in {year}.",
        "negative_tpl": "I'm not sure about {company}'s {attr} in {year}.",
    },
    {
        "template_id": "A2",
        "context_tpl": "{person} has served as the CEO of {company} since {year}. Under their leadership, {attr} reached {value}.",
        "question_tpl": "What is {company}'s {attr} under CEO {person}?",
        "answer_tpl": "{value}",
        "positive_tpl": "Under CEO {person}, {company}'s {attr} is {value}.",
        "negative_tpl": "I don't have that information about {company}'s {attr} under {person}.",
    },
    {
        "template_id": "A3",
        "context_tpl": "According to the {year} industry report, {company} achieved {attr} of {value} at its {location} facility.",
        "question_tpl": "At {company}'s {location} facility, what was the {attr} in {year}?",
        "answer_tpl": "{value}",
        "positive_tpl": "At its {location} facility, {company} had {attr} of {value} in {year}.",
        "negative_tpl": "I cannot confirm {company}'s {attr} at the {location} facility for {year}.",
    },
    {
        "template_id": "A4",
        "context_tpl": "{person}, the CTO of {company} based in {location}, announced that {attr} hit {value} in {year}.",
        "question_tpl": "Who announced {company}'s {attr} of {value} in {year} and where is the company based?",
        "answer_tpl": "{person} in {location}",
        "positive_tpl": "{person}, CTO of {company} based in {location}, announced that {attr} hit {value} in {year}.",
        "negative_tpl": "I don't recall who announced {company}'s {attr} figure.",
    },
    {
        "template_id": "A5",
        "context_tpl": "In {year}, {company} relocated its headquarters from {location} to {location2}. That same year, {attr} was {value}.",
        "question_tpl": "Where did {company} move its headquarters in {year} and what was its {attr}?",
        "answer_tpl": "{location2}; {attr} was {value}",
        "positive_tpl": "{company} moved its headquarters to {location2} in {year}, and its {attr} was {value}.",
        "negative_tpl": "I'm uncertain about {company}'s relocation details and its {attr}.",
    },
]

TEMPLATES_UNANSWERABLE = [
    {
        "template_id": "U1",
        "context_tpl": "{company} is a prominent firm in the {location} region. It has been operating since {year}.",
        "question_tpl": "What was {company}'s {attr} in {year_unrelated}?",
        "answer_tpl": None,
        "positive_tpl": "{company}'s {attr} in {year_unrelated} was {fake_value}.",  # model should NOT say this
        "negative_tpl": "The provided information does not contain data about {company}'s {attr} in {year_unrelated}.",
    },
    {
        "template_id": "U2",
        "context_tpl": "{person} joined {company} as a senior engineer in {year}. The office is located in {location}.",
        "question_tpl": "How many patents does {person} hold at {company}?",
        "answer_tpl": None,
        "positive_tpl": "{person} holds {fake_value} patents at {company}.",  # model should NOT say this
        "negative_tpl": "The context does not mention any patents held by {person} at {company}.",
    },
    {
        "template_id": "U3",
        "context_tpl": "{company}'s main product line includes cloud storage and data analytics tools.",
        "question_tpl": "What was {company}'s {attr} for Q3 of {year_unrelated}?",
        "answer_tpl": None,
        "positive_tpl": "In Q3 of {year_unrelated}, {company}'s {attr} was {fake_value}.",
        "negative_tpl": "The provided text does not include {company}'s financial data for that period.",
    },
    {
        "template_id": "U4",
        "context_tpl": "The city of {location} is known for its tech startup scene. {company} was founded there.",
        "question_tpl": "What is {person}'s role at {company} and when did they join?",
        "answer_tpl": None,
        "positive_tpl": "{person} is the {fake_role} at {company} and joined in {fake_year}.",
        "negative_tpl": "There is no information about {person}'s role at {company} in the given context.",
    },
    {
        "template_id": "U5",
        "context_tpl": "{company} received a series A funding round in {year}, led by an undisclosed investor.",
        "question_tpl": "How much total funding has {company} raised across all rounds?",
        "answer_tpl": None,
        "positive_tpl": "{company} has raised a total of ${fake_value}M across all funding rounds.",
        "negative_tpl": "The available information only mentions a series A round in {year}; total funding is not disclosed.",
    },
]


def _render(template: str, **kwargs) -> str:
    """Render a template string, leaving unmatched placeholders as-is."""
    for key, val in kwargs.items():
        template = template.replace("{" + key + "}", str(val))
    return template


def generate_samples(
    num_answerable: int,
    num_unanswerable: int,
    entity_pool_companies: List[str],
    entity_pool_people: List[str],
    entity_pool_locations: List[str],
    start_id: int = 0,
) -> List[Dict]:
    """Generate a list of synthetic QA samples."""

    samples = []
    entity_id = start_id

    templates_a = random.choices(TEMPLATES_ANSWERABLE, k=num_answerable)
    for tpl in templates_a:
        company = random.choice(entity_pool_companies)
        person = random.choice(entity_pool_people)
        location = random.choice(entity_pool_locations)
        location2 = random.choice([l for l in entity_pool_locations if l != location])
        year = random.choice(FAKE_YEARS)
        attr = random.choice(FAKE_ATTRIBUTES)
        value = random.randint(10, 990)

        kv = {
            "company": company,
            "person": person,
            "location": location,
            "location2": location2,
            "year": year,
            "attr": attr,
            "value": value,
        }

        context = _render(tpl["context_tpl"], **kv)
        question = _render(tpl["question_tpl"], **kv)
        gold_answer = _render(tpl["answer_tpl"], **kv)
        positive_response = _render(tpl["positive_tpl"], **kv)
        negative_response = _render(tpl["negative_tpl"], **kv)

        samples.append({
            "context": context,
            "question": question,
            "gold_answer": gold_answer,
            "answerability": "answerable",
            "positive_response": positive_response,
            "negative_response": negative_response,
            "entity_id": entity_id,
            "template_id": tpl["template_id"],
        })
        entity_id += 1

    templates_u = random.choices(TEMPLATES_UNANSWERABLE, k=num_unanswerable)
    for tpl in templates_u:
        company = random.choice(entity_pool_companies)
        person = random.choice(entity_pool_people)
        location = random.choice(entity_pool_locations)
        year = random.choice(FAKE_YEARS)
        year_unrelated = random.choice([y for y in FAKE_YEARS if abs(y - year) > 2])
        attr = random.choice(FAKE_ATTRIBUTES)
        fake_value = random.randint(10, 990)
        fake_year = random.choice(FAKE_YEARS)
        fake_role = random.choice(["VP of Engineering", "Head of Marketing", "Chief Architect", "Director of Operations"])

        kv = {
            "company": company,
            "person": person,
            "location": location,
            "year": year,
            "year_unrelated": year_unrelated,
            "attr": attr,
            "fake_value": fake_value,
            "fake_year": fake_year,
            "fake_role": fake_role,
        }

        context = _render(tpl["context_tpl"], **kv)
        question = _render(tpl["question_tpl"], **kv)
        gold_answer = None
        positive_response = _render(tpl["positive_tpl"], **kv)
        negative_response = _render(tpl["negative_tpl"], **kv)

        samples.append({
            "context": context,
            "question": question,
            "gold_answer": gold_answer,
            "answerability": "unanswerable",
            "positive_response": positive_response,
            "negative_response": negative_response,
            "entity_id": entity_id,
            "template_id": tpl["template_id"],
        })
        entity_id += 1

    random.shuffle(samples)
    return samples


def build_dataset(config: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Build train and test datasets with non-overlapping entity pools.
    config: dict with keys train_size, test_size, train_path, test_path.
    """

    train_size = config.get("train_size", 100)
    test_size = config.get("test_size", 100)

    half = len(FAKE_COMPANIES) // 2
    train_companies = FAKE_COMPANIES[:half]
    test_companies = FAKE_COMPANIES[half:]

    half_p = len(FAKE_PEOPLE) // 2
    train_people = FAKE_PEOPLE[:half_p]
    test_people = FAKE_PEOPLE[half_p:]

    half_l = len(FAKE_LOCATIONS) // 2
    train_locations = FAKE_LOCATIONS[:half_l]
    test_locations = FAKE_LOCATIONS[half_l:]

    num_a = train_size // 2
    num_u = train_size - num_a
    train = generate_samples(num_a, num_u, train_companies, train_people, train_locations, start_id=0)

    num_a_test = test_size // 2
    num_u_test = test_size - num_a_test
    test = generate_samples(num_a_test, num_u_test, test_companies, test_people, test_locations, start_id=5000)

    return train, test


def save_jsonl(samples: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


if __name__ == "__main__":
    random.seed(42)
    config = {"train_size": 100, "test_size": 100}
    train, test = build_dataset(config)
    save_jsonl(train, "data/train.jsonl")
    save_jsonl(test, "data/test.jsonl")
    print(f"Generated {len(train)} train samples, {len(test)} test samples.")