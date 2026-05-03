# Detekce vozidel z ortofota – Olomouc 2016

Semestrální práce z dálkového průzkumu Země.  
Oblast: Neředín / Nová Ulice, Olomouc.

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

## Skripty

```bash
# 1. Detekce vozidel z ortofota
python src/detect.py --conf 0.25 --slice-size 640 --overlap 0.2

# 2. Prostorová analýza + grafy
python src/analyze.py
```

Závislosti: `pip install geopandas scipy matplotlib` (+ ultralytics, sahi pro detekci)

---

## Limity a možná rozšíření

- **Jeden časový snímek** – parkování se v průběhu dne mění
- **Model natrénovaný na DOTA** – ne ideální pro česká sídliště, může plést stíny budov s auty
- **Voronoi ≠ skutečné spádové oblasti** – nevzohledňuje zástavbu, ulice ani vzdálenost chůze
- Možné rozšíření: kernel density estimation (KDE) místo Voronoi, nebo síťová analýza (nejbližší adresa po silnici)
