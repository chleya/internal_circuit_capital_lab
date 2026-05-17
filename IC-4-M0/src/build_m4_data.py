"""
IC-4-M4-Generalization: Generate test data for robustness validation.

Creates for each seed:
  - test_large: 120 samples (2x the M3 test size)
  - test_hard:  120 samples with extreme OOD (train entity pool = 1/4 of full pool)
"""

import json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data_builder import (
    FAKE_COMPANIES, FAKE_PEOPLE, FAKE_LOCATIONS,
    generate_samples, save_jsonl, load_jsonl,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data_m4")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [0, 1, 2]


def generate_test_set(seed, train_companies, train_people, train_locations,
                      test_companies, test_people, test_locations,
                      test_size, train_size, prefix):
    random.seed(seed)

    num_a = test_size // 2
    num_u = test_size - num_a
    test = generate_samples(num_a, num_u, test_companies, test_people,
                            test_locations, start_id=seed * 10000)

    num_a_train = train_size // 2
    num_u_train = train_size - num_a_train
    train = generate_samples(num_a_train, num_u_train, train_companies, train_people,
                             train_locations, start_id=seed * 10000 + 5000)

    test_path = os.path.join(OUT_DIR, f"{prefix}_s{seed}.jsonl")
    train_path = os.path.join(OUT_DIR, f"train_{prefix}_s{seed}.jsonl")
    save_jsonl(test, test_path)
    save_jsonl(train, train_path)

    na_test = sum(1 for s in test if s.get("answerability") == "answerable")
    na_train = sum(1 for s in train if s.get("answerability") == "answerable")
    print(f"  {prefix} seed={seed}: train {na_train}A+{len(train)-na_train}U, test {na_test}A+{len(test)-na_test}U")
    print(f"    saved to {test_path}, {train_path}")


for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")

    half_c = len(FAKE_COMPANIES) // 2
    half_p = len(FAKE_PEOPLE) // 2
    half_l = len(FAKE_LOCATIONS) // 2

    # ===== test_large: standard OOD split, 120 test samples =====
    print("  [test_large] standard OOD, 120 samples")
    generate_test_set(
        seed=seed,
        train_companies=FAKE_COMPANIES[:half_c],
        train_people=FAKE_PEOPLE[:half_p],
        train_locations=FAKE_LOCATIONS[:half_l],
        test_companies=FAKE_COMPANIES[half_c:],
        test_people=FAKE_PEOPLE[half_p:],
        test_locations=FAKE_LOCATIONS[half_l:],
        test_size=120,
        train_size=60,
        prefix="test_large",
    )

    # ===== test_hard: extreme OOD, train only 1/4 of entity pool =====
    print("  [test_hard] extreme OOD (train=1/4 pool), 120 samples")
    quarter_c = max(len(FAKE_COMPANIES) // 4, 4)
    quarter_p = max(len(FAKE_PEOPLE) // 4, 4)
    quarter_l = max(len(FAKE_LOCATIONS) // 4, 4)

    train_c = FAKE_COMPANIES[:quarter_c]
    test_c = FAKE_COMPANIES[quarter_c:]
    train_p = FAKE_PEOPLE[:quarter_p]
    test_p = FAKE_PEOPLE[quarter_p:]
    train_l = FAKE_LOCATIONS[:quarter_l]
    test_l = FAKE_LOCATIONS[quarter_l:]

    generate_test_set(
        seed=seed,
        train_companies=train_c,
        train_people=train_p,
        train_locations=train_l,
        test_companies=test_c,
        test_people=test_p,
        test_locations=test_l,
        test_size=120,
        train_size=30,
        prefix="test_hard",
    )

print("\nDone!")