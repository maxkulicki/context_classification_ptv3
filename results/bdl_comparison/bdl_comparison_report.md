# BDL vs Point Cloud Species Comparison

**Question**: How well does BDL cover the species actually present in our point cloud plots?

BDL describes entire forest subdivisions (potentially tens of hectares), while our plots are ~500 m² circles. BDL listing extra species is expected. The interesting cases are when **PC has a genus that BDL doesn't mention at all**.

## Dataset

- **BDL plots**: 272
- **PC plots**: 271
- **Overlapping**: 271
- **Genera**: 11 (Abies, Acer, Alnus, Betula, Carpinus, Fagus, Larix, Picea, Pinus, Quercus, Tilia)

## 1. Presence / Absence Overview

![Presence/Absence](presence_absence.png)

| Genus | Both | BDL-only | PC-only |
|-------|------|----------|---------|
| Abies | 32 | 30 | 0 |
| Acer | 23 | 60 | 3 |
| Alnus | 14 | 41 | 0 |
| Betula | 72 | 156 | 4 |
| Carpinus | 21 | 40 | 4 |
| Fagus | 63 | 67 | 3 |
| Larix | 20 | 58 | 2 |
| Picea | 102 | 105 | 0 |
| Pinus | 192 | 53 | 1 |
| Quercus | 85 | 99 | 3 |
| Tilia | 14 | 34 | 3 |

## 2. BDL Coverage of PC Species

For each genus, of the plots where our point cloud contains it, what fraction also has it listed in BDL?

![BDL Coverage](bdl_coverage.png)

| Genus | PC plots | BDL also lists | Coverage |
|-------|----------|---------------|----------|
| Abies | 32 | 32 | 100% |
| Acer | 26 | 23 | 88% |
| Alnus | 14 | 14 | 100% |
| Betula | 76 | 72 | 95% |
| Carpinus | 25 | 21 | 84% |
| Fagus | 66 | 63 | 95% |
| Larix | 22 | 20 | 91% |
| Picea | 102 | 102 | 100% |
| Pinus | 193 | 192 | 99% |
| Quercus | 88 | 85 | 97% |
| Tilia | 17 | 14 | 82% |

## 3. PC-only Cases: How Many Trees?

When the point cloud has a genus that BDL doesn't list, is it a single stray tree or multiple?

![PC-only counts](pc_only_counts.png)

| Genus | Plots | Total trees | Median per plot | Max per plot |
|-------|-------|-------------|----------------|-------------|
| Acer | 3 | 5 | 1 | 3 |
| Betula | 4 | 10 | 2 | 6 |
| Carpinus | 4 | 8 | 2 | 3 |
| Fagus | 3 | 4 | 1 | 2 |
| Larix | 2 | 3 | 2 | 2 |
| Pinus | 1 | 46 | 46 | 46 |
| Quercus | 3 | 5 | 1 | 3 |
| Tilia | 3 | 4 | 1 | 2 |

---

*BDL-only occurrences (BDL lists a genus not found in PC) are expected because subdivisions are much larger than our ~500 m² plots.*
