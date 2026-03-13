# Dataset Overview: Multi-Country TLS Tree Species Classification

**Prepared:** March 2026

---

## Summary Table

| Dataset | Country | Plots | Trees | Species | Scanner | Reference |
|---------|---------|-------|-------|---------|---------|-----------|
| **TreeScanPL** | Poland | 271 | 6,845 | 18 | TLS (Riegl VZ-400i) | Stereńczak et al. (in review) |
| **BioDiv-3DTrees** | Germany | 27 | 4,952 | 19 | TLS + ULS (Riegl VZ-400 / Phoenix Recon-XT) | Griese et al. (2025), *Scientific Data* |
| **LAUTx** | Austria | 6 | 434 | 6 | PLS (GeoSLAM ZEB Horizon) | Tockner et al. (2022), Zenodo |
| **Weiser et al.** | Germany | 12 | 264 | 12 | TLS (Riegl VZ-400), manual segmentation | Weiser et al. (2022), *ESSD* 14, 2989–3012 |
| **NIBIO** | Norway | 20 | 481 | 3 | ULS (Riegl miniVUX-1UAV) | Puliti et al. (2023), Zenodo (FOR-Instance) |
| **CULS** | Czech Republic | — | 50 | 1 | ULS | Puliti et al. (2023), Zenodo (FOR-Instance) |
| **Frey 2022** | Germany | 15 | 472 | 6 | TLS | Frey et al. (2022); ForSpecies-GPS |
| **Junttila / Yrttimaa** | Finland | 20 | 51 | 1 | ULS | Junttila et al. / Yrttimaa et al.; ForSpecies-GPS |
| **Puliti MLS** | Italy (Tuscany) | 1 | 67 | 1 | MLS | Puliti et al.; ForSpecies-GPS |
| **Puliti ULS 2** | Norway/Finland | — | 621 | 3 | ULS | Puliti et al.; ForSpecies-GPS |
| **Saarinen 2021** | Finland | 10 | 1,976 | 1 | MLS | Saarinen et al. (2021); ForSpecies-GPS |
| **Wytham Woods** | UK (Oxford) | 1 | 769 | 6 | TLS | ForSpecies-GPS |
| **Total** | **9 countries** | **406+** | **16,982** | — | — | — |

---

## TreeScanPL (Poland)

Core dataset. 271 circular plots (15 m radius) with complete TLS point clouds from Polish national forests, spanning multiple forest districts across lowland and upland regions. Trees manually segmented and labeled with species from field inventory. Georeferenced — all contextual features (AlphaEarth, SNIR, GeoPlantNet, BDL) available. Dominant species is Scots pine (52%), reflecting the species composition of Polish managed forests.

| Species | Count | % |
|---------|-------|---|
| Pinus sylvestris | 3,565 | 52.1 |
| Picea abies | 907 | 13.3 |
| Fagus sylvatica | 489 | 7.1 |
| Quercus sp. | 467 | 6.8 |
| Abies alba | 416 | 6.1 |
| Betula pendula | 415 | 6.1 |
| Carpinus betulus | 125 | 1.8 |
| Larix decidua | 106 | 1.5 |
| Alnus glutinosa | 82 | 1.2 |
| Tilia cordata | 77 | 1.1 |
| Quercus rubra | 51 | 0.7 |
| Acer pseudoplatanus | 41 | 0.6 |
| Fraxinus excelsior | 21 | 0.3 |
| Alnus incana | 20 | 0.3 |
| Prunus avium | 19 | 0.3 |
| Acer platanoides | 17 | 0.2 |
| Populus tremula | 15 | 0.2 |
| Sorbus aucuparia | 12 | 0.2 |

---

## BioDiv-3DTrees (Germany)

**Griese, N. et al. (2025).** "A large dataset of labelled single tree point clouds, QSMs and tree graphs." *Scientific Data*. DOI: 10.1038/s41597-025-06421-7

4,952 individually segmented trees from 27 plots across the three German Biodiversity Exploratories (DFG SPP 1374): Schwäbische Alb (AEW, 2,046 trees), Hainich-Dün (HEW, 1,532), and Schorfheide-Chorin (SEW, 1,374). TLS acquired with Riegl VZ-400 in leaf-off condition; ULS from DJI M300 RTK + Phoenix Recon-XT at 75 m AGL (604 pts/m²). Segmented using TreeLearn with quality control against existing forest inventory. In addition to point clouds, the dataset provides QSM reconstructions for 3,386 broadleaved trees and tree graph representations. Species labels matched from the Biodiversity Exploratories forest inventory database. Exact plot coordinates are sensitive and must be requested through BExIS — this affects extraction of geolocation-dependent features (AlphaEarth, SNIR, GeoPlantNet).

| Species | Count | % |
|---------|-------|---|
| Fagus sylvatica | 2,992 | 60.4 |
| Picea abies | 1,005 | 20.3 |
| Pinus sylvestris | 512 | 10.3 |
| Quercus spec. | 178 | 3.6 |
| Carpinus betulus | 66 | 1.3 |
| Acer pseudoplatanus | 65 | 1.3 |
| Fraxinus excelsior | 59 | 1.2 |
| Betula spec. | 24 | 0.5 |
| Pseudotsuga menziesii | 11 | 0.2 |
| Larix decidua | 10 | 0.2 |
| Tilia spec. | 10 | 0.2 |
| 8 other species | 20 | 0.4 |

---

## LAUTx (Austria)

**Tockner, A., Gollob, C., Ritter, T. & Nothdurft, A. (2022).** "LAUTx - Individual Tree Point Clouds from Austrian Forest Inventory Plots." Zenodo. DOI: 10.5281/zenodo.6560112

**Companion:** Gollob, C. et al. (2020). Raw PLS point clouds: LAUT dataset. DOI: 10.5281/zenodo.3698956

Six circular plots (~1,257 m² each) in Lower Austrian mixed forests, scanned with a GeoSLAM ZEB Horizon handheld PLS in March 2019 (leaf-off). Trees manually segmented, originally created as a tree segmentation benchmark. Labels later extended/corrected by Henrich et al. (2024) with propagation to full plot-level point clouds. Plots are mostly single-layered with moderate to dense understory. The PLS scan pattern differs significantly from stationary TLS — lower angular resolution, SLAM-based registration, different noise characteristics. One of the first published datasets with fully manual 3D instance annotations for forest point clouds.

| Species | Count | % |
|---------|-------|---|
| Fagus sylvatica (be) | 262 | 60.4 |
| Picea abies (sp) | 123 | 28.3 |
| Larix decidua (la) | 29 | 6.7 |
| Pinus sylvestris (pin) | 8 | 1.8 |
| Abies alba (fir) | 7 | 1.6 |
| Quercus sp. (oak) | 5 | 1.2 |
| *81 trees unlabeled — excluded* | | |

---

## Weiser et al. 2022 (Germany)

**Weiser, H., Schäfer, J., Winiwarter, L., Krašovec, N., Fassnacht, F.E. & Höfle, B. (2022).** "Individual tree point clouds and tree measurements from multi-platform laser scanning in German forests." *Earth System Science Data*, 14, 2989–3012. DOI: 10.5194/essd-14-2989-2022

**Data:** PANGAEA, DOI: 10.1594/PANGAEA.942856

Multi-platform dataset from 12 plots in two mixed temperate forest areas near Bretten and Karlsruhe (Baden-Württemberg, SW Germany), plus one plot near Speyer. The original dataset provides spatially overlapping, georeferenced point clouds from TLS (Riegl VZ-400), ULS (Riegl VUX-1LR on DJI M600 Pro), and ALS (Riegl VQ-780i). We use only the TLS subset with manual segmentation. Coordinates in ETRS89/UTM32N. The well-balanced species distribution (no species >18%) and species-level oak labels (Q. petraea, Q. robur, Q. rubra) make it particularly valuable for classification despite its small size.

| Species | Count | % |
|---------|-------|---|
| Pseudotsuga menziesii | 47 | 17.8 |
| Fagus sylvatica | 46 | 17.4 |
| Pinus sylvestris | 36 | 13.6 |
| Quercus petraea | 33 | 12.5 |
| Picea abies | 30 | 11.4 |
| Acer pseudoplatanus | 28 | 10.6 |
| Quercus rubra | 23 | 8.7 |
| Abies alba | 9 | 3.4 |
| Carpinus betulus | 6 | 2.3 |
| Quercus robur | 4 | 1.5 |
| Larix decidua | 1 | 0.4 |
| Tsuga heterophylla | 1 | 0.4 |

---

## NIBIO / FOR-Instance (Norway)

**Puliti, S., Pearse, G., Surovy, P., et al. (2023).** "FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees." Zenodo. DOI: 10.5281/zenodo.8287792

**See also:** Puliti, S. et al. (2025). "Benchmarking tree species classification from proximally sensed laser scanning data: Introducing the FOR-species20K dataset." *Methods in Ecology and Evolution*, 16, 801–818. DOI: 10.1111/2041-210X.14503

The NIBIO subset of FOR-Instance: 20 plots in boreal coniferous forests in southeastern Norway, acquired by ULS (Riegl miniVUX-1UAV). Trees manually annotated with instance IDs and semantic labels. Dominated by Norway spruce (81%) with minor Scots pine and birch. ULS provides good canopy coverage but sparser stem detail compared to TLS. The broader FOR-Instance dataset includes plots from Czech Republic, Austria, New Zealand, and Australia; extended to FOR-InstanceV2 (Xiang et al., 2025) with additional regions. Only the original NIBIO ULS subset with species labels is used here as the boreal/Scandinavian geographic anchor for cross-country generalization testing.

| Species | Count | % |
|---------|-------|---|
| Picea abies | 391 | 81.3 |
| Betula sp. | 51 | 10.6 |
| Pinus sylvestris | 39 | 8.1 |

---

## CULS / FOR-Instance (Czech Republic)

**Puliti, S., Pearse, G., Surovy, P., et al. (2023).** "FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees." Zenodo. DOI: 10.5281/zenodo.8287792

The CULS subset of FOR-Instance, contributed by Czech University of Life Sciences in Prague. 50 Pinus sylvestris trees from Czech coniferous forests, acquired by ULS. Single-species dataset — provides an additional Central European Scots pine sample alongside the NIBIO subset from Norway.

| Species | Count | % |
|---------|-------|---|
| Pinus sylvestris | 50 | 100.0 |

---

## Frey 2022 (Germany)

Part of the ForSpecies-GPS dataset (a subset of ForSpecies that includes geolocations, not part of the official ForSpecies release). TLS dataset with 472 individually segmented trees from 15 locations (B1–B5, K1–K5, S1–S5) in German forests, each with 3 trees per subplot. Well-balanced species mix with a strong representation of Pseudotsuga menziesii (33%) and Quercus robur (24%). Provides species-level oak identification (Q. robur), unlike some datasets that use genus-level labels.

| Species | Count | % |
|---------|-------|---|
| Pseudotsuga menziesii | 157 | 33.3 |
| Quercus robur | 113 | 23.9 |
| Picea abies | 75 | 15.9 |
| Pinus sylvestris | 68 | 14.4 |
| Abies alba | 45 | 9.5 |
| Larix decidua | 14 | 3.0 |

---

## Junttila / Yrttimaa (Finland)

Part of the ForSpecies-GPS dataset. ULS dataset from 20 plots in Finnish boreal forests. Contains only Betula pendula (51 trees). Single-species dataset — useful for cross-dataset consistency checks on birch geometric features rather than species classification.

| Species | Count | % |
|---------|-------|---|
| Betula pendula | 51 | 100.0 |

---

## Puliti MLS (Italy)

Part of the ForSpecies-GPS dataset. MLS dataset from a single plot in Tuscany, Italy (43.76°N, 11.57°E). Contains only Abies alba (67 trees). Point clouds are not georeferenced but plot coordinates are provided. Single-species dataset — contributes additional Abies alba samples from a southern European provenance.

| Species | Count | % |
|---------|-------|---|
| Abies alba | 67 | 100.0 |

---

## Puliti ULS 2 (Norway/Finland)

Part of the ForSpecies-GPS dataset. ULS dataset with 621 trees across 3 boreal species. Dominated by Betula pendula (54%) and Pinus sylvestris (39%) with minor Picea abies. Provides a substantial additional pool of boreal species from ULS acquisition.

| Species | Count | % |
|---------|-------|---|
| Betula pendula | 335 | 54.0 |
| Pinus sylvestris | 241 | 38.8 |
| Picea abies | 45 | 7.2 |

---

## Saarinen 2021 (Finland)

Part of the ForSpecies-GPS dataset. MLS dataset from 10 plots in Finnish forests containing 1,976 Pinus sylvestris trees. The largest single-species contribution in the collection. Single-species dataset — valuable for evaluating cross-scanner and cross-geography consistency of Scots pine geometric features.

| Species | Count | % |
|---------|-------|---|
| Pinus sylvestris | 1,976 | 100.0 |

---

## Wytham Woods (UK)

Part of the ForSpecies-GPS dataset. TLS dataset from Wytham Woods, Oxford, UK (51.77°N, −1.34°W). Point clouds are not georeferenced but approximate stand coordinates are provided. 769 trees across 6 species, heavily dominated by Acer pseudoplatanus (72%). Uniquely contributes understory/hedgerow species not found in other datasets: Corylus avellana, Crataegus monogyna, and Acer campestre.

| Species | Count | % |
|---------|-------|---|
| Acer pseudoplatanus | 552 | 71.8 |
| Fraxinus excelsior | 85 | 11.1 |
| Corylus avellana | 67 | 8.7 |
| Quercus robur | 37 | 4.8 |
| Crataegus monogyna | 26 | 3.4 |
| Acer campestre | 2 | 0.3 |

---

## Cross-Dataset Species Overlap

Species with ≥5 trees in at least one dataset. Bold = largest source for that species.

| Species | TreeScanPL | BioDiv3D | LAUTx | Weiser | NIBIO | CULS | Frey | Junttila | Puliti MLS | Puliti ULS 2 | Saarinen | Wytham |
|---------|-----------|----------|-------|--------|-------|------|------|----------|------------|-------------|----------|--------|
| Pinus sylvestris | **3,565** | 512 | 8 | 36 | 39 | 50 | 68 | — | — | 241 | **1,976** | — |
| Picea abies | 907 | **1,005** | 123 | 30 | 391 | — | 75 | — | — | 45 | — | — |
| Fagus sylvatica | 489 | **2,992** | **262** | 46 | — | — | — | — | — | — | — | — |
| Quercus sp.† | 467 | 178 | 5 | 60 | — | — | 113 | — | — | — | — | 37 |
| Abies alba | **416** | 5 | 7 | 9 | — | — | 45 | — | 67 | — | — | — |
| Betula sp. | **415** | 24 | — | — | 51 | — | — | 51 | — | **335** | — | — |
| Acer pseudoplatanus | 41 | 65 | — | 28 | — | — | — | — | — | — | — | **552** |
| Pseudotsuga menziesii | — | 11 | — | 47 | — | — | **157** | — | — | — | — | — |
| Carpinus betulus | 125 | **66** | — | 6 | — | — | — | — | — | — | — | — |
| Larix decidua | **106** | 10 | 29 | 1 | — | — | 14 | — | — | — | — | — |
| Fraxinus excelsior | 21 | **59** | — | — | — | — | — | — | — | — | — | 85 |
| Alnus glutinosa | **82** | — | — | — | — | — | — | — | — | — | — | — |
| Tilia cordata | **77** | 10 | — | — | — | — | — | — | — | — | — | — |
| Quercus rubra | **51** | 1 | — | 23 | — | — | — | — | — | — | — | — |
| Corylus avellana | — | — | — | — | — | — | — | — | — | — | — | **67** |
| Crataegus monogyna | — | — | — | — | — | — | — | — | — | — | — | **26** |

†Quercus: TreeScanPL has Q. sp. and Q. rubra separately; BioDiv3D has Q. spec.; Weiser has Q. petraea (33), Q. rubra (23), Q. robur (4); Frey has Q. robur; Wytham has Q. robur.

## Total Trees per Species

| Species | Total | Datasets |
|---------|------:|----------|
| Pinus sylvestris | 6,495 | 8 |
| Fagus sylvatica | 3,789 | 4 |
| Picea abies | 2,576 | 7 |
| Betula sp. | 876 | 5 |
| Quercus sp.† | 860 | 6 |
| Acer pseudoplatanus | 686 | 4 |
| Abies alba | 549 | 6 |
| Pseudotsuga menziesii | 215 | 3 |
| Carpinus betulus | 197 | 3 |
| Fraxinus excelsior | 165 | 3 |
| Larix decidua | 160 | 5 |
| Tilia cordata | 87 | 2 |
| Alnus glutinosa | 82 | 1 |
| Quercus rubra | 75 | 3 |
| Corylus avellana | 67 | 1 |
| Crataegus monogyna | 26 | 1 |
| Acer campestre | 2 | 1 |

†Aggregates all Quercus entries except Q. rubra (listed separately).
