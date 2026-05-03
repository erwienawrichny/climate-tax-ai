# ================================================================
# GOOGLE DATA CENTER PUE + REAL WEATHER DATA COLLECTION
# PUE source: datacenters.google/efficiency (verified)
# Weather source: Open-Meteo archive API (free, no key needed)
# Run: python data_collection_final.py
# Output: google_datacenter_pue_final.csv
# ================================================================

import requests
import pandas as pd
import numpy as np
import time

# ================================================================
# STEP 1: VERIFIED PUE DATA
# All values directly from Google's efficiency page
# 2022-2025 full quarterly data
# "Lowcountry, South Carolina" standardized to "Berkeley County, SC"
# ================================================================

pue_data = [
    # ── 2022 Q1 ──
    ("Douglas County, Georgia",        2022,"Q1",1.08),("Lenoir, North Carolina",         2022,"Q1",1.08),("Berkeley County, South Carolina", 2022,"Q1",1.08),("Montgomery County, Tennessee",   2022,"Q1",1.11),("Jackson County, Alabama",         2022,"Q1",1.10),("Loudoun County, Virginia",        2022,"Q1",1.07),("Loudoun County, Virginia (2nd)",  2022,"Q1",1.08),("New Albany, Ohio",                2022,"Q1",1.14),("Council Bluffs, Iowa",            2022,"Q1",1.11),("Council Bluffs, Iowa (2nd)",      2022,"Q1",1.07),("Papillion, Nebraska",             2022,"Q1",1.18),("Mayes County, Oklahoma",          2022,"Q1",1.08),("Midlothian, Texas",               2022,"Q1",1.16),("The Dalles, Oregon",              2022,"Q1",1.10),("The Dalles, Oregon (2nd)",        2022,"Q1",1.08),("Henderson, Nevada",               2022,"Q1",1.09),("Dublin, Ireland",                 2022,"Q1",1.09),("St. Ghislain, Belgium",           2022,"Q1",1.08),("Eemshaven, Netherlands",          2022,"Q1",1.06),("Fredericia, Denmark",             2022,"Q1",1.13),("Hamina, Finland",                 2022,"Q1",1.09),("Changhua County, Taiwan",         2022,"Q1",1.10),("Singapore",                       2022,"Q1",1.13),("Singapore (2nd)",                 2022,"Q1",1.23),("Quilicura, Chile",                2022,"Q1",1.10),
    # ── 2022 Q2 ──
    ("Douglas County, Georgia",        2022,"Q2",1.10),("Lenoir, North Carolina",         2022,"Q2",1.09),("Berkeley County, South Carolina", 2022,"Q2",1.10),("Montgomery County, Tennessee",   2022,"Q2",1.12),("Jackson County, Alabama",         2022,"Q2",1.12),("Loudoun County, Virginia",        2022,"Q2",1.09),("Loudoun County, Virginia (2nd)",  2022,"Q2",1.09),("New Albany, Ohio",                2022,"Q2",1.13),("Council Bluffs, Iowa",            2022,"Q2",1.12),("Council Bluffs, Iowa (2nd)",      2022,"Q2",1.08),("Papillion, Nebraska",             2022,"Q2",1.12),("Mayes County, Oklahoma",          2022,"Q2",1.11),("Midlothian, Texas",               2022,"Q2",1.18),("The Dalles, Oregon",              2022,"Q2",1.09),("The Dalles, Oregon (2nd)",        2022,"Q2",1.07),("Henderson, Nevada",               2022,"Q2",1.10),("Dublin, Ireland",                 2022,"Q2",1.09),("St. Ghislain, Belgium",           2022,"Q2",1.09),("Eemshaven, Netherlands",          2022,"Q2",1.07),("Fredericia, Denmark",             2022,"Q2",1.11),("Hamina, Finland",                 2022,"Q2",1.09),("Changhua County, Taiwan",         2022,"Q2",1.13),("Singapore",                       2022,"Q2",1.14),("Singapore (2nd)",                 2022,"Q2",1.23),("Quilicura, Chile",                2022,"Q2",1.09),
    # ── 2022 Q3 ──
    ("Douglas County, Georgia",        2022,"Q3",1.12),("Lenoir, North Carolina",         2022,"Q3",1.11),("Berkeley County, South Carolina", 2022,"Q3",1.13),("Montgomery County, Tennessee",   2022,"Q3",1.14),("Jackson County, Alabama",         2022,"Q3",1.14),("Loudoun County, Virginia",        2022,"Q3",1.11),("Loudoun County, Virginia (2nd)",  2022,"Q3",1.11),("New Albany, Ohio",                2022,"Q3",1.18),("Council Bluffs, Iowa",            2022,"Q3",1.14),("Council Bluffs, Iowa (2nd)",      2022,"Q3",1.10),("Papillion, Nebraska",             2022,"Q3",1.13),("Mayes County, Oklahoma",          2022,"Q3",1.13),("Midlothian, Texas",               2022,"Q3",1.19),("The Dalles, Oregon",              2022,"Q3",1.10),("The Dalles, Oregon (2nd)",        2022,"Q3",1.08),("Henderson, Nevada",               2022,"Q3",1.15),("Storey County, Nevada",           2022,"Q3",1.29),("Dublin, Ireland",                 2022,"Q3",1.09),("St. Ghislain, Belgium",           2022,"Q3",1.09),("Eemshaven, Netherlands",          2022,"Q3",1.08),("Fredericia, Denmark",             2022,"Q3",1.11),("Hamina, Finland",                 2022,"Q3",1.09),("Changhua County, Taiwan",         2022,"Q3",1.14),("Singapore",                       2022,"Q3",1.13),("Singapore (2nd)",                 2022,"Q3",1.21),("Quilicura, Chile",                2022,"Q3",1.09),
    # ── 2022 Q4 ──
    ("Douglas County, Georgia",        2022,"Q4",1.07),("Lenoir, North Carolina",         2022,"Q4",1.08),("Berkeley County, South Carolina", 2022,"Q4",1.08),("Montgomery County, Tennessee",   2022,"Q4",1.08),("Jackson County, Alabama",         2022,"Q4",1.10),("Loudoun County, Virginia",        2022,"Q4",1.07),("Loudoun County, Virginia (2nd)",  2022,"Q4",1.07),("New Albany, Ohio",                2022,"Q4",1.13),("Council Bluffs, Iowa",            2022,"Q4",1.10),("Council Bluffs, Iowa (2nd)",      2022,"Q4",1.07),("Papillion, Nebraska",             2022,"Q4",1.10),("Mayes County, Oklahoma",          2022,"Q4",1.07),("Midlothian, Texas",               2022,"Q4",1.12),("The Dalles, Oregon",              2022,"Q4",1.09),("The Dalles, Oregon (2nd)",        2022,"Q4",1.05),("Henderson, Nevada",               2022,"Q4",1.09),("Storey County, Nevada",           2022,"Q4",1.20),("Dublin, Ireland",                 2022,"Q4",1.09),("St. Ghislain, Belgium",           2022,"Q4",1.09),("Eemshaven, Netherlands",          2022,"Q4",1.07),("Fredericia, Denmark",             2022,"Q4",1.12),("Hamina, Finland",                 2022,"Q4",1.09),("Changhua County, Taiwan",         2022,"Q4",1.11),("Singapore",                       2022,"Q4",1.13),("Singapore (2nd)",                 2022,"Q4",1.19),("Quilicura, Chile",                2022,"Q4",1.09),
    # ── 2023 Q1 ──
    ("Douglas County, Georgia",        2023,"Q1",1.07),("Lenoir, North Carolina",         2023,"Q1",1.08),("Berkeley County, South Carolina", 2023,"Q1",1.08),("Montgomery County, Tennessee",   2023,"Q1",1.10),("Jackson County, Alabama",         2023,"Q1",1.09),("Loudoun County, Virginia",        2023,"Q1",1.07),("Loudoun County, Virginia (2nd)",  2023,"Q1",1.07),("New Albany, Ohio",                2023,"Q1",1.13),("Council Bluffs, Iowa",            2023,"Q1",1.10),("Council Bluffs, Iowa (2nd)",      2023,"Q1",1.07),("Papillion, Nebraska",             2023,"Q1",1.12),("Mayes County, Oklahoma",          2023,"Q1",1.08),("Midlothian, Texas",               2023,"Q1",1.11),("The Dalles, Oregon",              2023,"Q1",1.09),("The Dalles, Oregon (2nd)",        2023,"Q1",1.07),("Henderson, Nevada",               2023,"Q1",1.07),("Storey County, Nevada",           2023,"Q1",1.18),("Dublin, Ireland",                 2023,"Q1",1.09),("St. Ghislain, Belgium",           2023,"Q1",1.09),("Eemshaven, Netherlands",          2023,"Q1",1.07),("Fredericia, Denmark",             2023,"Q1",1.12),("Hamina, Finland",                 2023,"Q1",1.09),("Changhua County, Taiwan",         2023,"Q1",1.09),("Singapore",                       2023,"Q1",1.13),("Singapore (2nd)",                 2023,"Q1",1.19),("Quilicura, Chile",                2023,"Q1",1.09),
    # ── 2023 Q2 ──
    ("Douglas County, Georgia",        2023,"Q2",1.09),("Lenoir, North Carolina",         2023,"Q2",1.09),("Berkeley County, South Carolina", 2023,"Q2",1.11),("Montgomery County, Tennessee",   2023,"Q2",1.10),("Jackson County, Alabama",         2023,"Q2",1.11),("Loudoun County, Virginia",        2023,"Q2",1.08),("Loudoun County, Virginia (2nd)",  2023,"Q2",1.08),("New Albany, Ohio",                2023,"Q2",1.10),("Council Bluffs, Iowa",            2023,"Q2",1.11),("Council Bluffs, Iowa (2nd)",      2023,"Q2",1.08),("Papillion, Nebraska",             2023,"Q2",1.09),("Mayes County, Oklahoma",          2023,"Q2",1.10),("Midlothian, Texas",               2023,"Q2",1.14),("The Dalles, Oregon",              2023,"Q2",1.10),("The Dalles, Oregon (2nd)",        2023,"Q2",1.07),("Henderson, Nevada",               2023,"Q2",1.08),("Storey County, Nevada",           2023,"Q2",1.20),("Dublin, Ireland",                 2023,"Q2",1.08),("St. Ghislain, Belgium",           2023,"Q2",1.08),("Eemshaven, Netherlands",          2023,"Q2",1.07),("Fredericia, Denmark",             2023,"Q2",1.10),("Hamina, Finland",                 2023,"Q2",1.09),("Changhua County, Taiwan",         2023,"Q2",1.13),("Singapore",                       2023,"Q2",1.13),("Singapore (2nd)",                 2023,"Q2",1.19),("Quilicura, Chile",                2023,"Q2",1.09),
    # ── 2023 Q3 ──
    ("Douglas County, Georgia",        2023,"Q3",1.13),("Lenoir, North Carolina",         2023,"Q3",1.12),("Berkeley County, South Carolina", 2023,"Q3",1.14),("Montgomery County, Tennessee",   2023,"Q3",1.13),("Jackson County, Alabama",         2023,"Q3",1.13),("Loudoun County, Virginia",        2023,"Q3",1.10),("Loudoun County, Virginia (2nd)",  2023,"Q3",1.11),("New Albany, Ohio",                2023,"Q3",1.11),("Council Bluffs, Iowa",            2023,"Q3",1.14),("Council Bluffs, Iowa (2nd)",      2023,"Q3",1.09),("Papillion, Nebraska",             2023,"Q3",1.10),("Mayes County, Oklahoma",          2023,"Q3",1.14),("Midlothian, Texas",               2023,"Q3",1.16),("The Dalles, Oregon",              2023,"Q3",1.10),("The Dalles, Oregon (2nd)",        2023,"Q3",1.08),("Henderson, Nevada",               2023,"Q3",1.10),("Storey County, Nevada",           2023,"Q3",1.24),("Dublin, Ireland",                 2023,"Q3",1.08),("St. Ghislain, Belgium",           2023,"Q3",1.09),("Eemshaven, Netherlands",          2023,"Q3",1.09),("Fredericia, Denmark",             2023,"Q3",1.08),("Hamina, Finland",                 2023,"Q3",1.10),("Changhua County, Taiwan",         2023,"Q3",1.15),("Singapore",                       2023,"Q3",1.13),("Singapore (2nd)",                 2023,"Q3",1.19),("Quilicura, Chile",                2023,"Q3",1.09),
    # ── 2023 Q4 ──
    ("Douglas County, Georgia",        2023,"Q4",1.08),("Lenoir, North Carolina",         2023,"Q4",1.08),("Berkeley County, South Carolina", 2023,"Q4",1.08),("Montgomery County, Tennessee",   2023,"Q4",1.09),("Jackson County, Alabama",         2023,"Q4",1.09),("Loudoun County, Virginia",        2023,"Q4",1.09),("Loudoun County, Virginia (2nd)",  2023,"Q4",1.07),("New Albany, Ohio",                2023,"Q4",1.08),("Council Bluffs, Iowa",            2023,"Q4",1.10),("Council Bluffs, Iowa (2nd)",      2023,"Q4",1.07),("Papillion, Nebraska",             2023,"Q4",1.08),("Mayes County, Oklahoma",          2023,"Q4",1.08),("Midlothian, Texas",               2023,"Q4",1.10),("The Dalles, Oregon",              2023,"Q4",1.10),("The Dalles, Oregon (2nd)",        2023,"Q4",1.06),("Henderson, Nevada",               2023,"Q4",1.07),("Storey County, Nevada",           2023,"Q4",1.14),("Dublin, Ireland",                 2023,"Q4",1.08),("St. Ghislain, Belgium",           2023,"Q4",1.08),("Eemshaven, Netherlands",          2023,"Q4",1.07),("Fredericia, Denmark",             2023,"Q4",1.10),("Hamina, Finland",                 2023,"Q4",1.09),("Changhua County, Taiwan",         2023,"Q4",1.11),("Singapore",                       2023,"Q4",1.13),("Singapore (2nd)",                 2023,"Q4",1.17),("Quilicura, Chile",                2023,"Q4",1.09),
    # ── 2024 Q1 ──
    ("Berkeley County, South Carolina",2024,"Q1",1.08),("Changhua County, Taiwan",        2024,"Q1",1.10),("Council Bluffs, Iowa",            2024,"Q1",1.12),("Council Bluffs, Iowa (2nd)",      2024,"Q1",1.07),("Douglas County, Georgia",         2024,"Q1",1.07),("Dublin, Ireland",                 2024,"Q1",1.08),("Eemshaven, Netherlands",          2024,"Q1",1.07),("Fredericia, Denmark",             2024,"Q1",1.09),("Hamina, Finland",                 2024,"Q1",1.09),("Henderson, Nevada",               2024,"Q1",1.07),("Jackson County, Alabama",         2024,"Q1",1.08),("Lenoir, North Carolina",          2024,"Q1",1.08),("Loudoun County, Virginia",        2024,"Q1",1.08),("Loudoun County, Virginia (2nd)",  2024,"Q1",1.07),("Mayes County, Oklahoma",          2024,"Q1",1.08),("Midlothian, Texas",               2024,"Q1",1.09),("Montgomery County, Tennessee",    2024,"Q1",1.09),("New Albany, Ohio",                2024,"Q1",1.08),("Papillion, Nebraska",             2024,"Q1",1.09),("Quilicura, Chile",                2024,"Q1",1.11),("Singapore",                       2024,"Q1",1.13),("Singapore (2nd)",                 2024,"Q1",1.16),("St. Ghislain, Belgium",           2024,"Q1",1.08),("Storey County, Nevada",           2024,"Q1",1.12),("The Dalles, Oregon",              2024,"Q1",1.10),("The Dalles, Oregon (2nd)",        2024,"Q1",1.06),
    # ── 2024 Q2 ──
    ("Berkeley County, South Carolina",2024,"Q2",1.11),("Changhua County, Taiwan",        2024,"Q2",1.13),("Council Bluffs, Iowa",            2024,"Q2",1.11),("Council Bluffs, Iowa (2nd)",      2024,"Q2",1.07),("Douglas County, Georgia",         2024,"Q2",1.10),("Dublin, Ireland",                 2024,"Q2",1.08),("Eemshaven, Netherlands",          2024,"Q2",1.08),("Fredericia, Denmark",             2024,"Q2",1.08),("Hamina, Finland",                 2024,"Q2",1.10),("Henderson, Nevada",               2024,"Q2",1.09),("Jackson County, Alabama",         2024,"Q2",1.11),("Lenoir, North Carolina",          2024,"Q2",1.11),("Loudoun County, Virginia",        2024,"Q2",1.09),("Loudoun County, Virginia (2nd)",  2024,"Q2",1.08),("Mayes County, Oklahoma",          2024,"Q2",1.12),("Midlothian, Texas",               2024,"Q2",1.12),("Montgomery County, Tennessee",    2024,"Q2",1.11),("New Albany, Ohio",                2024,"Q2",1.09),("Papillion, Nebraska",             2024,"Q2",1.09),("Quilicura, Chile",                2024,"Q2",1.09),("Singapore",                       2024,"Q2",1.13),("Singapore (2nd)",                 2024,"Q2",1.15),("St. Ghislain, Belgium",           2024,"Q2",1.08),("Storey County, Nevada",           2024,"Q2",1.16),("The Dalles, Oregon",              2024,"Q2",1.10),("The Dalles, Oregon (2nd)",        2024,"Q2",1.06),
    # ── 2024 Q3 ──
    ("Berkeley County, South Carolina",2024,"Q3",1.13),("Central Ohio (Columbus), Ohio",  2024,"Q3",1.05),("Central Ohio (Lancaster), Ohio",  2024,"Q3",1.04),("Central Ohio (New Albany), Ohio", 2024,"Q3",1.07),("Changhua County, Taiwan",         2024,"Q3",1.15),("Council Bluffs, Iowa",            2024,"Q3",1.13),("Council Bluffs, Iowa (2nd)",      2024,"Q3",1.08),("Douglas County, Georgia",         2024,"Q3",1.12),("Dublin, Ireland",                 2024,"Q3",1.07),("Eemshaven, Netherlands",          2024,"Q3",1.08),("Fredericia, Denmark",             2024,"Q3",1.07),("Hamina, Finland",                 2024,"Q3",1.10),("Henderson, Nevada",               2024,"Q3",1.11),("Jackson County, Alabama",         2024,"Q3",1.13),("Lenoir, North Carolina",          2024,"Q3",1.24),("Loudoun County, Virginia",        2024,"Q3",1.11),("Loudoun County, Virginia (2nd)",  2024,"Q3",1.09),("Mayes County, Oklahoma",          2024,"Q3",1.14),("Midlothian, Texas",               2024,"Q3",1.13),("Montgomery County, Tennessee",    2024,"Q3",1.13),("Papillion, Nebraska",             2024,"Q3",1.10),("Quilicura, Chile",                2024,"Q3",1.09),("Singapore",                       2024,"Q3",1.13),("Singapore (2nd)",                 2024,"Q3",1.15),("St. Ghislain, Belgium",           2024,"Q3",1.08),("Storey County, Nevada",           2024,"Q3",1.22),("The Dalles, Oregon",              2024,"Q3",1.11),("The Dalles, Oregon (2nd)",        2024,"Q3",1.08),
    # ── 2024 Q4 ──
    ("Berkeley County, South Carolina",2024,"Q4",1.08),("Central Ohio (Columbus), Ohio",  2024,"Q4",1.05),("Central Ohio (Lancaster), Ohio",  2024,"Q4",1.04),("Central Ohio (New Albany), Ohio", 2024,"Q4",1.05),("Changhua County, Taiwan",         2024,"Q4",1.12),("Council Bluffs, Iowa",            2024,"Q4",1.09),("Council Bluffs, Iowa (2nd)",      2024,"Q4",1.06),("Douglas County, Georgia",         2024,"Q4",1.08),("Dublin, Ireland",                 2024,"Q4",1.08),("Eemshaven, Netherlands",          2024,"Q4",1.07),("Fredericia, Denmark",             2024,"Q4",1.07),("Hamina, Finland",                 2024,"Q4",1.09),("Henderson, Nevada",               2024,"Q4",1.08),("Inzai, Japan",                    2024,"Q4",1.15),("Jackson County, Alabama",         2024,"Q4",1.09),("Lenoir, North Carolina",          2024,"Q4",1.08),("Loudoun County, Virginia",        2024,"Q4",1.07),("Loudoun County, Virginia (2nd)",  2024,"Q4",1.07),("Mayes County, Oklahoma",          2024,"Q4",1.09),("Midlothian, Texas",               2024,"Q4",1.08),("Montgomery County, Tennessee",    2024,"Q4",1.08),("Papillion, Nebraska",             2024,"Q4",1.08),("Quilicura, Chile",                2024,"Q4",1.09),("Singapore",                       2024,"Q4",1.13),("Singapore (2nd)",                 2024,"Q4",1.14),("St. Ghislain, Belgium",           2024,"Q4",1.07),("Storey County, Nevada",           2024,"Q4",1.11),("The Dalles, Oregon",              2024,"Q4",1.10),("The Dalles, Oregon (2nd)",        2024,"Q4",1.06),
    # ── 2025 Q1 ──
    ("Berkeley County, South Carolina",2025,"Q1",1.07),("Central Ohio (Columbus), Ohio",  2025,"Q1",1.05),("Central Ohio (Lancaster), Ohio",  2025,"Q1",1.04),("Central Ohio (New Albany), Ohio", 2025,"Q1",1.05),("Changhua County, Taiwan",         2025,"Q1",1.10),("Council Bluffs, Iowa",            2025,"Q1",1.10),("Council Bluffs, Iowa (2nd)",      2025,"Q1",1.07),("Douglas County, Georgia",         2025,"Q1",1.08),("Dublin, Ireland",                 2025,"Q1",1.08),("Eemshaven, Netherlands",          2025,"Q1",1.07),("Fredericia, Denmark",             2025,"Q1",1.07),("Hamina, Finland",                 2025,"Q1",1.09),("Henderson, Nevada",               2025,"Q1",1.07),("Inzai, Japan",                    2025,"Q1",1.10),("Jackson County, Alabama",         2025,"Q1",1.09),("Lenoir, North Carolina",          2025,"Q1",1.08),("Loudoun County, Virginia",        2025,"Q1",1.07),("Loudoun County, Virginia (2nd)",  2025,"Q1",1.07),("Mayes County, Oklahoma",          2025,"Q1",1.09),("Midlothian, Texas",               2025,"Q1",1.08),("Montgomery County, Tennessee",    2025,"Q1",1.08),("Papillion, Nebraska",             2025,"Q1",1.08),("Quilicura, Chile",                2025,"Q1",1.09),("Singapore",                       2025,"Q1",1.12),("Singapore (2nd)",                 2025,"Q1",1.14),("St. Ghislain, Belgium",           2025,"Q1",1.07),("Storey County, Nevada",           2025,"Q1",1.09),("The Dalles, Oregon",              2025,"Q1",1.10),("The Dalles, Oregon (2nd)",        2025,"Q1",1.06),
    # ── 2025 Q2 ──
    ("Berkeley County, South Carolina",2025,"Q2",1.11),("Central Ohio (Columbus), Ohio",  2025,"Q2",1.05),("Central Ohio (Lancaster), Ohio",  2025,"Q2",1.04),("Central Ohio (New Albany), Ohio", 2025,"Q2",1.07),("Changhua County, Taiwan",         2025,"Q2",1.14),("Council Bluffs, Iowa",            2025,"Q2",1.11),("Council Bluffs, Iowa (2nd)",      2025,"Q2",1.08),("Douglas County, Georgia",         2025,"Q2",1.10),("Dublin, Ireland",                 2025,"Q2",1.08),("Eemshaven, Netherlands",          2025,"Q2",1.07),("Fredericia, Denmark",             2025,"Q2",1.06),("Hamina, Finland",                 2025,"Q2",1.09),("Henderson, Nevada",               2025,"Q2",1.09),("Inzai, Japan",                    2025,"Q2",1.14),("Jackson County, Alabama",         2025,"Q2",1.11),("Lenoir, North Carolina",          2025,"Q2",1.10),("Loudoun County, Virginia",        2025,"Q2",1.08),("Loudoun County, Virginia (2nd)",  2025,"Q2",1.08),("Mayes County, Oklahoma",          2025,"Q2",1.16),("Midlothian, Texas",               2025,"Q2",1.11),("Montgomery County, Tennessee",    2025,"Q2",1.11),("Papillion, Nebraska",             2025,"Q2",1.09),("Quilicura, Chile",                2025,"Q2",1.08),("Singapore",                       2025,"Q2",1.13),("Singapore (2nd)",                 2025,"Q2",1.15),("St. Ghislain, Belgium",           2025,"Q2",1.08),("Storey County, Nevada",           2025,"Q2",1.16),("The Dalles, Oregon",              2025,"Q2",1.10),("The Dalles, Oregon (2nd)",        2025,"Q2",1.06),
    # ── 2025 Q3 ──
    ("Berkeley County, South Carolina",2025,"Q3",1.12),("Central Ohio (Columbus), Ohio",  2025,"Q3",1.07),("Central Ohio (Lancaster), Ohio",  2025,"Q3",1.04),("Central Ohio (New Albany), Ohio", 2025,"Q3",1.07),("Changhua County, Taiwan",         2025,"Q3",1.16),("Council Bluffs, Iowa",            2025,"Q3",1.14),("Council Bluffs, Iowa (2nd)",      2025,"Q3",1.10),("Douglas County, Georgia",         2025,"Q3",1.12),("Dublin, Ireland",                 2025,"Q3",1.08),("Eemshaven, Netherlands",          2025,"Q3",1.08),("Fredericia, Denmark",             2025,"Q3",1.06),("Hamina, Finland",                 2025,"Q3",1.10),("Henderson, Nevada",               2025,"Q3",1.12),("Inzai, Japan",                    2025,"Q3",1.14),("Jackson County, Alabama",         2025,"Q3",1.13),("Lenoir, North Carolina",          2025,"Q3",1.12),("Loudoun County, Virginia",        2025,"Q3",1.10),("Loudoun County, Virginia (2nd)",  2025,"Q3",1.10),("Mayes County, Oklahoma",          2025,"Q3",1.15),("Midlothian, Texas",               2025,"Q3",1.13),("Montgomery County, Tennessee",    2025,"Q3",1.10),("Papillion, Nebraska",             2025,"Q3",1.11),("Quilicura, Chile",                2025,"Q3",1.08),("Singapore",                       2025,"Q3",1.12),("Singapore (2nd)",                 2025,"Q3",1.15),("St. Ghislain, Belgium",           2025,"Q3",1.08),("Storey County, Nevada",           2025,"Q3",1.22),("The Dalles, Oregon",              2025,"Q3",1.11),("The Dalles, Oregon (2nd)",        2025,"Q3",1.08),
    # ── 2025 Q4 ──
    ("Berkeley County, South Carolina",2025,"Q4",1.08),("Central Ohio (Columbus), Ohio",  2025,"Q4",1.05),("Central Ohio (Lancaster), Ohio",  2025,"Q4",1.04),("Central Ohio (New Albany), Ohio", 2025,"Q4",1.06),("Changhua County, Taiwan",         2025,"Q4",1.12),("Council Bluffs, Iowa",            2025,"Q4",1.09),("Council Bluffs, Iowa (2nd)",      2025,"Q4",1.07),("Douglas County, Georgia",         2025,"Q4",1.07),("Dublin, Ireland",                 2025,"Q4",1.08),("Eemshaven, Netherlands",          2025,"Q4",1.07),("Fredericia, Denmark",             2025,"Q4",1.06),("Hamina, Finland",                 2025,"Q4",1.09),("Henderson, Nevada",               2025,"Q4",1.08),("Inzai, Japan",                    2025,"Q4",1.07),("Jackson County, Alabama",         2025,"Q4",1.09),("Lenoir, North Carolina",          2025,"Q4",1.08),("Loudoun County, Virginia",        2025,"Q4",1.07),("Loudoun County, Virginia (2nd)",  2025,"Q4",1.07),("Mayes County, Oklahoma",          2025,"Q4",1.09),("Midlothian, Texas",               2025,"Q4",1.09),("Montgomery County, Tennessee",    2025,"Q4",1.06),("Papillion, Nebraska",             2025,"Q4",1.07),("Quilicura, Chile",                2025,"Q4",1.08),("Singapore",                       2025,"Q4",1.12),("Singapore (2nd)",                 2025,"Q4",1.14),("St. Ghislain, Belgium",           2025,"Q4",1.07),("Storey County, Nevada",           2025,"Q4",1.11),("The Dalles, Oregon",              2025,"Q4",1.10),("The Dalles, Oregon (2nd)",        2025,"Q4",1.06),
]

# ================================================================
# STEP 2: COORDINATES
# ================================================================

coordinates = {
    "Berkeley County, South Carolina":  (33.19, -79.96),
    "Central Ohio (Columbus), Ohio":    (39.96, -82.99),
    "Central Ohio (Lancaster), Ohio":   (39.71, -82.59),
    "Central Ohio (New Albany), Ohio":  (40.08, -82.79),
    "Changhua County, Taiwan":          (24.05, 120.52),
    "Council Bluffs, Iowa":             (41.26, -95.86),
    "Council Bluffs, Iowa (2nd)":       (41.26, -95.86),
    "Douglas County, Georgia":          (33.70, -84.70),
    "Dublin, Ireland":                  (53.33,  -6.25),
    "Eemshaven, Netherlands":           (53.44,   6.83),
    "Fredericia, Denmark":              (55.57,   9.75),
    "Hamina, Finland":                  (60.57,  27.20),
    "Henderson, Nevada":                (36.03,-114.98),
    "Inzai, Japan":                     (35.83, 140.15),
    "Jackson County, Alabama":          (34.74, -85.97),
    "Lenoir, North Carolina":           (35.91, -81.54),
    "Loudoun County, Virginia":         (39.08, -77.64),
    "Loudoun County, Virginia (2nd)":   (39.08, -77.64),
    "Mayes County, Oklahoma":           (36.30, -95.24),
    "Midlothian, Texas":                (32.48, -97.00),
    "Montgomery County, Tennessee":     (36.50, -87.35),
    "New Albany, Ohio":                 (40.08, -82.79),
    "Papillion, Nebraska":              (41.15, -96.04),
    "Quilicura, Chile":                 (-33.36,-70.73),
    "Singapore":                        (  1.35, 103.82),
    "Singapore (2nd)":                  (  1.35, 103.82),
    "St. Ghislain, Belgium":            (50.45,   3.82),
    "Storey County, Nevada":            (39.52,-119.52),
    "The Dalles, Oregon":               (45.59,-121.18),
    "The Dalles, Oregon (2nd)":         (45.59,-121.18),
}

# ================================================================
# STEP 3: ELECTRICITY PRICES (business rate $/kWh)
# ================================================================

facility_elec = {
    "Berkeley County, South Carolina":  ("USA",         0.148),
    "Central Ohio (Columbus), Ohio":    ("USA",         0.148),
    "Central Ohio (Lancaster), Ohio":   ("USA",         0.148),
    "Central Ohio (New Albany), Ohio":  ("USA",         0.148),
    "Changhua County, Taiwan":          ("Taiwan",      0.187),
    "Council Bluffs, Iowa":             ("USA",         0.148),
    "Council Bluffs, Iowa (2nd)":       ("USA",         0.148),
    "Douglas County, Georgia":          ("USA",         0.148),
    "Dublin, Ireland":                  ("Ireland",     0.270),
    "Eemshaven, Netherlands":           ("Netherlands", 0.220),
    "Fredericia, Denmark":              ("Denmark",     0.234),
    "Hamina, Finland":                  ("Finland",     0.124),
    "Henderson, Nevada":                ("USA",         0.148),
    "Inzai, Japan":                     ("Japan",       0.202),
    "Jackson County, Alabama":          ("USA",         0.148),
    "Lenoir, North Carolina":           ("USA",         0.148),
    "Loudoun County, Virginia":         ("USA",         0.148),
    "Loudoun County, Virginia (2nd)":   ("USA",         0.148),
    "Mayes County, Oklahoma":           ("USA",         0.148),
    "Midlothian, Texas":                ("USA",         0.148),
    "Montgomery County, Tennessee":     ("USA",         0.148),
    "New Albany, Ohio":                 ("USA",         0.148),
    "Papillion, Nebraska":              ("USA",         0.148),
    "Quilicura, Chile":                 ("Chile",       0.166),
    "Singapore":                        ("Singapore",   0.265),
    "Singapore (2nd)":                  ("Singapore",   0.265),
    "St. Ghislain, Belgium":            ("Belgium",     0.261),
    "Storey County, Nevada":            ("USA",         0.148),
    "The Dalles, Oregon":               ("USA",         0.148),
    "The Dalles, Oregon (2nd)":         ("USA",         0.148),
}

# ================================================================
# STEP 4: QUARTER DATE RANGES
# ================================================================

quarter_dates = {
    "Q1": ("01-01", "03-31"),
    "Q2": ("04-01", "06-30"),
    "Q3": ("07-01", "09-30"),
    "Q4": ("10-01", "12-31"),
}

# ================================================================
# STEP 5: FETCH REAL WEATHER FROM OPEN-METEO
# Free, no API key needed
# Returns actual observed temperature and humidity per quarter
# ================================================================

weather_cache = {}

def get_weather(facility, year, quarter):
    key = (facility, year, quarter)
    if key in weather_cache:
        return weather_cache[key]

    lat, lon = coordinates[facility]
    start_m, end_m = quarter_dates[quarter]
    start_date = f"{year}-{start_m}"
    end_date   = f"{year}-{end_m}"

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      "temperature_2m_mean,relative_humidity_2m_mean",
        "timezone":   "auto",
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        temps = [t for t in daily.get("temperature_2m_mean",       []) if t is not None]
        hums  = [h for h in daily.get("relative_humidity_2m_mean", []) if h is not None]
        result = (
            round(sum(temps) / len(temps), 2) if temps else None,
            round(sum(hums)  / len(hums),  2) if hums  else None,
        )
    except Exception as e:
        print(f"  ⚠ Weather fetch failed: {facility} {year} {quarter} — {e}")
        result = (None, None)

    weather_cache[key] = result
    time.sleep(0.1)
    return result

# ================================================================
# STEP 6: BUILD COMBINED DATASET
# ================================================================

print(f"Fetching real weather data for {len(pue_data)} facility-quarter combinations...")
print("Takes ~3-4 minutes. Progress updates every 50 rows.\n")

rows = []
for i, (facility, year, quarter, pue) in enumerate(pue_data):
    temp, humidity = get_weather(facility, year, quarter)
    country, elec  = facility_elec[facility]

    rows.append({
        "facility":               facility,
        "year":                   year,
        "quarter":                quarter,
        "pue":                    pue,
        "avg_temp_c":             temp,
        "avg_humidity_pct":       humidity,
        "latitude":               coordinates[facility][0],
        "longitude":              coordinates[facility][1],
        "country":                country,
        "electricity_business_usd": elec,
    })

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(pue_data)} rows fetched...")

df = pd.DataFrame(rows)
df_clean = df.dropna(subset=["avg_temp_c", "avg_humidity_pct"])

# ================================================================
# STEP 7: SAVE
# ================================================================

output = "google_datacenter_pue_final.csv"
df_clean.to_csv(output, index=False)

print(f"\n✅ Done!")
print(f"   Total rows:        {len(df)}")
print(f"   Rows with weather: {len(df_clean)}")
print(f"   Facilities:        {df_clean['facility'].nunique()}")
print(f"   Years:             {sorted(df_clean['year'].unique())}")
print(f"\nPreview:")
print(df_clean.head(8).to_string(index=False))
print(f"\nSaved to: {output}")

# Auto-download if running in Google Colab
try:
    from google.colab import files
    files.download(output)
    print("Download started in your browser.")
except ImportError:
    pass
