from __future__ import annotations

SCHEMA_VERSION = "1.0.0"
CAMPAIGN_ID = "biosnap-dti-v2"
DEFAULT_DATASET_VERSION = "biosnap-dti-canonical-v2"
DEFAULT_SPLIT_SEED = 41
DEFAULT_SEEDS = (13, 29, 71)

PSEUDO_ENTITIES = frozenset({"#Drug", "Gene"})

BIOSNAP_URL = "https://snap.stanford.edu/biodata/datasets/10002/files/ChG-Miner_miner-chem-gene.tsv.gz"
EXPECTED_SOURCE_SHA256 = "b54a548bb0b6d7039b5c317bf8251d17233bdc89bc49f477aeb8abddfafc9e6c"
EXPECTED_EDGES = 15_138
EXPECTED_DRUGS = 5_017
EXPECTED_PROTEINS = 2_324
EXPECTED_ENTITIES = 7_341

FINAL_TEST_TOKEN = "OPEN_SEALED_TEST_ONCE"
HEURISTIC_METHODS = frozenset({"degree", "resource_allocation", "three_hop"})
TRAINED_METHODS = frozenset({"skipgnn", "gcn", "graphsage", "residual_three_hop"})
COMBINE_METHODS = frozenset({"blend_pathlens_three_hop", "rrf_pathlens_three_hop"})
DIAGNOSTIC_METHODS = frozenset({"ranking_diagnostics"})
IMPORT_METHODS = frozenset(
    {"one_hop", "s1_s2", "s1_s2_s3_fixed", "pathlens_bce", "pathlens_ranking"}
)
STAGES = ("smoke", "train", "eval", "final")
V2_FREEZE_CHECKPOINT_SHA256 = (
    "646700d456ba4cb0de795e7067e1830e8f18598b97fe5cc89d195d631d356586"
)
