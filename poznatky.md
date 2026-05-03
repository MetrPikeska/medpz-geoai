# Detekce a analýza vozidel z ortofota – Olomouc 2016

**Semestrální práce** | Pokročilé metody dálkového průzkumu Země (MEDPZ)  
**Autoři:** Petr Mikeska, Daniel Lee Číp  
**Oblast:** Neředín / Nová Ulice, Olomouc

---

## Co jsme dělali

1. **Detekce aut z ortofota** pomocí neuronové sítě (YOLOv8-OBB + SAHI tiling)
2. **Voronoi zóny** z adresních bodů – každá adresa dostala svůj polygon „nejbližšího okolí"
3. **Prostorové spojení** – přiřazení detekovaných vozidel do Voronoi zón
4. **Statistická analýza** – hustota vozidel na obyvatele a korelace s věkovými skupinami

---

## Data

| Dataset | Zdroj | Popis |
|---|---|---|
| Ortofoto | Olomouc 2016 | RGB snímek, rozlišení ~10 cm/px, EPSG:5514 |
| Adresní body | RÚIAN (prosinec 2023) | 307 adresních míst s demografickými daty |
| Detekovaná vozidla | výstup detekce | GeoPackage, polygony aut v EPSG:5514 |

Demografická data v adresách: počet obyvatel podle věkových skupin (0–14, 15–29, 30–44, 45–64, 65+) a celkový počet.

---

## Metody

### Detekce vozidel

- Model: **YOLOv8-OBB** (Oriented Bounding Box) natrénovaný na datasetu DOTA
- Problém velkých snímků → **SAHI tiling**: snímek se rozřeže na překrývající se dlaždice, detekce proběhne na každé zvlášť, výsledky se sloučí
- Parametry: confidence = 0.25, tile = 640 px, overlap = 20 %

### Voronoi polygony

- Ze 307 adresních bodů jsme vytvořili 307 Voronoi polygonů
- Každý polygon = oblast, která je geometricky nejblíže dané adrese
- Polygony oříznuty na bounding box studované oblasti (+ 100 m buffer)

### Analýza hustoty

- Pro každou Voronoi zónu: počet detekovaných vozidel, rozdělení na malá / velká
- **Hustota** = počet vozidel / celkový počet obyvatel adresy
- Korelace: Pearsonův r mezi podílem věkové skupiny a hustotou vozidel
- Filtr: pouze zóny s ≥ 10 obyvateli (162 ze 307)

---

## Výsledky

### Detekce vozidel

| | |
|---|---|
| Celkem detekovaných vozidel | **1 124** |
| Malá vozidla (osobní auta) | 830 (73,8 %) |
| Velká vozidla (dodávky, autobusy) | 294 (26,2 %) |
| Průměr vozidel na Voronoi zónu | 3,66 |
| Medián vozidel na zónu | 1,0 |
| Maximum v jedné zóně | **35 aut** (Jílová 264/2, Nová Ulice) |

Velký rozdíl průměr vs. medián ukazuje silnou pravostrannou asymetrii – většina adres má 0–2 auta, ale pár míst (parkoviště) tahá průměr nahoru.

### Hustota vozidel na obyvatele

| | |
|---|---|
| Průměrná hustota | 0,128 aut/os |
| Mediánová hustota | 0,083 aut/os |
| Maximum | **0,562 aut/os** (tř. Svornosti 891/44, Nová Ulice) |

Hustota ~0,13 odpovídá přibližně 1 autu na každých 8 obyvatel – realistické číslo pro jedno konkrétní parkovací místo u adresy, ne pro celkové vlastnictví aut.

### Korelace: věkové skupiny vs. hustota vozidel

Korelace počítána jako Pearsonův r mezi **podílem věkové skupiny** a **density_per_resident**.  
Filtr: zóny s ≥ 10 obyvateli (n = 162).

| Věková skupina | r | p | Signifikantní? |
|---|---|---|---|
| 0–14 | −0,128 | 0,1056 | ne |
| 15–29 | −0,086 | 0,2781 | ne |
| 30–44 | −0,098 | 0,2133 | ne |
| 45–64 | −0,005 | 0,9493 | ne |
| 65+ | +0,136 | 0,0846 | ne (hraničně) |

**Žádná věková skupina nepredikuje hustotu aut statisticky významně** (p > 0,05).

### Co to znamená

Dřívější analýza (absolutní počty aut vs. absolutní počty obyvatel) ukazovala silnou pozitivní korelaci r ≈ 0,5 pro všechny skupiny. **To byl artefakt**: větší bytovky mají přirozeně víc lidí i víc aut v okolí – šlo tedy jen o efekt velikosti budovy, ne o demografii.

Po přepočtu na hustotu (auta na osobu) korelace zmizela. Závěr: **věková struktura obyvatel adresy neovlivňuje počet zaparkovaných aut v jejím bezprostředním okolí** – alespoň ne měřitelně tímto způsobem.

Možné důvody:
- Vozidla na ortofotu nejsou „auta obyvatel dané adresy" – jsou to auta parkující v okolí (chodníky, ulice, parkoviště)
- Jeden snímek = jeden okamžik → nezachycuje denní variabilitu
- Prostorová přesnost Voronoi zón je hrubá (každý bod dostane zónu bez ohledu na skutečné parkovací možnosti)

---

## Výstupní soubory

| Soubor | Popis |
|---|---|
| `data/vectors/voronoi.gpkg` | GeoPackage pro QGIS – vrstvy: Voronoi zóny (s počty aut), vozidla, adresy |
| `outputs/analysis_map.png` | Mapa: počty aut + hustota na obyvatele |
| `outputs/correlations.png` | Bar chart Pearsonových r pro věkové skupiny |
| `outputs/scatter_seniors.png` | Scatter: podíl seniorů vs. auta/obyvatel |
| `outputs/statistics.csv` | Tabulka za každou adresu |
| `outputs/correlations.csv` | Korelační koeficienty |

---

## Evaluace kvality detekce (parametrický sweep)

Skript `src/evaluate.py` otestoval 27 kombinací parametrů na výřezu 2000×2000 px
centrovaném na oblast s nejvyšší hustotou vozidel.

### Co se sweepovalo

| Parametr | Hodnoty | Co ovlivňuje |
|---|---|---|
| `conf` | 0.15 / 0.25 / 0.35 | confidence threshold YOLOv8 |
| `overlap` | 0.1 / 0.2 / 0.3 | překryv dlaždic SAHI |
| `postprocess_match_threshold` | 0.3 / 0.5 / 0.7 | IOS práh pro sloučení boxů přes dlaždice |

### Výsledky: duplicity

Klíčový parametr je `postprocess_match_threshold` — práh IOS (Intersection over Smaller),
nad kterým SAHI sloučí dva boxy do jednoho:

| iou_match | Průměrný dup_rate |
|---|---|
| **0.3** | **0.0 %** |
| 0.5 | ~1.5 % |
| 0.7 | ~6 % |

`conf` a `overlap` duplicity téměř neovlivní — rozhoduje výhradně `iou_match`.

### Výsledky: klasifikace malá vs. velká vozidla

| Konfigurace | size_ratio (large/small) | Mann-Whitney p |
|---|---|---|
| všechny (27 konfigurací) | 0.97–1.05 | 0.06–0.65 |

**Klasifikace je nezávislá na nastavení parametrů.** Medián plochy boxu je ~10.5–11.5 m²
pro *obě třídy* bez ohledu na conf/overlap/iou_match. Nikdy není statisticky oddělená (p > 0.05).

**Příčina:** YOLOv8-OBB byl natrénován na DOTA (satelitní snímky, stovky metrů výška).
V DOTA se „small vehicle" vs. „large vehicle" rozlišuje jinak než na 10 cm/px ortofotu.
Model vidí jiné textury a úhly než na trénovacích datech → třídy spolehlivě neodlišuje.

### Doporučená konfigurace

```
conf=0.25  overlap=0.2  postprocess_match_threshold=0.3
```

- 0 % duplicit
- 124 vozidel na testovacím výřezu (vs. 125 při conf=0.25/overlap=0.2/iou=0.5)
- Tato hodnota je nyní výchozí v `src/detect.py`

**Klasifikaci small/large nelze zlepšit laděním parametrů** — vyžadovalo by to fine-tuning
modelu na ručně anotovaných datech z tohoto ortofota.

---

## Skripty

```bash
# 1. Detekce vozidel z ortofota
python src/detect.py --conf 0.25 --slice-size 640 --overlap 0.2

# 2. Prostorová analýza + grafy
python src/analyze.py

# 3. Rozšířená analýza parkování
python src/parking_analysis.py

# 4. Evaluace kvality detekce (analýza existujících dat)
python src/evaluate.py analyze

# 4b. Parametrický sweep (trvá ~20 min)
python src/evaluate.py sweep
```

Závislosti: `pip install geopandas scipy matplotlib scikit-learn sahi ultralytics rasterio`

---

## Limity a možná rozšíření

- **Jeden časový snímek** – parkování se v průběhu dne mění
- **Klasifikace small/large nespolehlivá** – model DOTA neodpovídá česky sídlišti; pro spolehlivou klasifikaci by bylo nutné fine-tunovat na ručních anotacích
- **Voronoi ≠ skutečné spádové oblasti** – nevzohledňuje zástavbu, ulice ani vzdálenost chůze
- Možné rozšíření: kernel density estimation (KDE) místo Voronoi, nebo síťová analýza (nejbližší adresa po silnici)
