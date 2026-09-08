# Jak analýza funguje — od fotky k číslu

Podrobný popis toho, co se v aplikaci se snímkem děje a **proč** to tak je.
Text je psaný tak, aby se dal číst od začátku do konce a člověk po něm rozuměl
nejen tomu, které tlačítko co dělá, ale i tomu, proč je metoda postavená právě
takhle a kde má hranice.

Všechny ukázky kódu jsou **doslovné výřezy ze zdrojáků** v tomto repozitáři
(soubor a funkce je vždy uvedená). Všechna čísla a obrázky vznikly spuštěním
skutečného analytického jádra — nejsou to schémata nakreslená ručně.

**Obsah**

1. [Fyzika: proč se v temném poli dá kontaminace vůbec měřit](#1-fyzika)
2. [Mapa celého řetězce](#2-mapa-celého-řetězce)
3. [Krok 1 — načtení a normalizace](#3-krok-1--načtení-a-normalizace)
4. [Krok 2 — geometrie: binning a výřez](#4-krok-2--geometrie-binning-a-výřez)
5. [Krok 3 — odečet referenčního pozadí](#5-krok-3--odečet-referenčního-pozadí)
6. [Krok 4 — rozklad na zamlžení a ostrou složku](#6-krok-4--rozklad-na-zamlžení-a-ostrou-složku)
7. [Krok 5 — odhad šumu a práh (nejdůležitější krok)](#7-krok-5--odhad-šumu-a-práh)
8. [Krok 6 — segmentace na objekty](#8-krok-6--segmentace-na-objekty)
9. [Krok 7 — klasifikace typu kontaminace](#9-krok-7--klasifikace-typu-kontaminace)
10. [Krok 8 — z masek na čísla](#10-krok-8--z-masek-na-čísla)
11. [Krok 9 — čas: derivace a fáze děje](#11-krok-9--čas-derivace-a-fáze-děje)
12. [Co která páčka dělá](#12-co-která-páčka-dělá)
13. [Limity metody — co to neumí](#13-limity-metody--co-to-neumí)
14. [Jak si to osahat](#14-jak-si-to-osahat)

---

## 1. Fyzika

V **temném poli** nejde přímé světlo do objektivu. Osvětlení je zařízené tak, že
kdyby bylo sklíčko dokonale čisté a hladké, kamera by viděla černou. Do objektivu
se dostane jen světlo, které se na něčem **rozptýlilo** — na částici prachu, na
vlákně, na kapce, na škrábanci.

To má tři důsledky, na kterých stojí celá analýza:

1. **Kontaminace jas jen přidává, nikdy neubírá.** Signál je jednostranný.
   Tenhle fakt se v kódu využívá na několika místech (odhad nuly, směr prahu,
   interpretace derivace).
2. **Jas ≈ míra rozptylu, ne velikost.** Malá, ale silně rozptylující částice
   může být jasnější než velké slabě rozptylující smítko. Proto se **primárně
   měří plocha**, ne jas — plocha je robustnější veličina.
3. **Citlivost je enormní.** Když je pozadí černé, i velmi slabý signál je vidět.
   Zároveň to znamená, že se stejně dobře „vidí“ i šum senzoru a prach usazený
   na optice mikroskopu. Bez odečtu pozadí je měření bezcenné.

Body 1 a 3 spolu tvoří základní úlohu, kterou analýza řeší: **oddělit to, co na
sklíčku přibylo, od toho, co tam (a v přístroji) bylo pořád.**

---

## 2. Mapa celého řetězce

![Řetězec zpracování jednoho snímku](img/01_retezec.png)

Obrázek je vyrobený skutečnou funkcí `analyze_prepared_frame()` nad snímkem,
jehož pozadí je **reálná reference ze 4K kamery** (`mean_mono` z `.npz`) a do
kterého bylo uměle vloženo 45 mikročástic, 3 shluky a 3 vlákna. Klasifikace
nakonec našla **42 mikročástic, 3 shluky a 3 vlákna** — tedy přesně to, co tam
bylo vloženo (tři nejslabší smítka zapadla pod práh).

Všimněte si panelů 1 a 2: **snímek a referenční pozadí vypadají skoro stejně.**
Ten „prach“ v levé horní části není kontaminace, kterou měříme — to je struktura
optiky a senzoru. Právě proto se odečítá. Až panel 3 ukazuje, co doopravdy
přibylo.

Řetězec v bodech:

| # | Krok | Vstup → výstup | Kde v kódu |
|---|------|----------------|------------|
| 1 | Načtení a normalizace | soubor → `float32` v ADU | `frameio.load_frame()` |
| 2 | Geometrie | plné rozlišení → binning + ROI | `analyzer.apply_geometry()` |
| 3 | Odečet pozadí | snímek − bias → diference | `analyzer.analyze_prepared_frame()` |
| 4 | Rozklad | diference → opar + ostrá složka | `analyzer.separate_haze()` |
| 5 | Odhad šumu a práh | ostrá složka → binární maska | `analyzer.estimate_noise()` |
| 6 | Segmentace | maska → očíslované objekty | `cv2.connectedComponentsWithStats` |
| 7 | Klasifikace | objekty → částice / shluk / vlákno | `analyzer.classify_components()` |
| 8 | Metriky | masky → ~40 čísel | `analyzer.analyze_prepared_frame()` |
| 9 | Časová řada | čísla → derivace a fáze | `analyzer.compute_rates_and_phases()` |

Kroky 1–8 běží pro každý snímek zvlášť a jsou nezávislé — proto se dají pustit
paralelně. Krok 9 potřebuje celou sérii najednou.

---

## 3. Krok 1 — načtení a normalizace

### Jednotná škála ADU

Kamera může ukládat 8 bitů (0–255), ale i 10, 12 nebo 16 bitů. Kdyby analýza
pracovala se surovými hodnotami, znamenal by „práh 4“ u 8bitové kamery něco
úplně jiného než u 12bitové. Všechno se proto převádí na jednotnou stupnici
**0–255 ADU** (*Analog-to-Digital Unit*, jednotka jasu po převodu z fotonů):

```python
    native_dtype = str(image.dtype)
    scale = float(full_scale) if full_scale else _guess_full_scale(image)
    scale = max(scale, 1.0)

    mode = normalize_mono_mode(mono_mode)
    data = np.asarray(to_mono(image, mode), dtype=np.float32)
    if abs(scale - DISPLAY_FULL_SCALE) > 1e-6:
        data = data * np.float32(DISPLAY_FULL_SCALE / scale)
```
<sub>`frameio.py`, funkce `load_frame()`</sub>

Dvě věci, které tu stojí za povšimnutí:

* Výsledek je **`float32`, ne `uint8`.** To je zásadní. V celočíselné aritmetice
  by `snímek − bias` u tmavších míst spadlo na nulu (saturace) a přišli bychom
  o zápornou půlku rozdělení — a přesně z ní se odhaduje šum. Podrobněji
  v [kroku 5](#7-krok-5--odhad-šumu-a-práh).
* `full_scale` se dá předat zvenčí. Při analýze série se zjistí jednou z prvních
  snímků a pak se používá pro všechny, takže se stupnice uprostřed měření
  nezmění.

### Barevný snímek → intenzita

Reference (`mean_mono`) je jednokanálová, takže barevný snímek se musí na jeden
kanál promítnout — a je jedno, jestli je kamera černobílá nebo barevná, dál už
řetězec pracuje s jedním číslem na pixel:

```python
    mode = normalize_mono_mode(mono_mode)
    if mode in ("b", "g", "r"):
        return image[:, :, {"b": 0, "g": 1, "r": 2}[mode]]

    planes = image.astype(np.float32, copy=False)
    if mode == "maximum":
        return planes.max(axis=2)
    if mode == "prumer":
        return planes.mean(axis=2, dtype=np.float32)
    return planes @ _LUMA_WEIGHTS          # luma (Rec.601, pořadí B, G, R)
```
<sub>`frameio.py`, funkce `to_mono()`</sub>

Výchozí je **luma podle Rec.601** (`0,299 R + 0,587 G + 0,114 B`), protože přesně
tak počítá černobílý obraz sama kamera — a tedy i referenci `mean_mono`. Kdyby se
použil jiný převod, reference by na snímky neseděla a rozdíl by se projevil jako
plošné zamlžení.

Ostatní režimy mají smysl, když víte, co děláte:

* **průměr kanálů** nepodceňuje modrou. V temném poli je Rayleighův rozptyl
  (∝ 1/λ⁴) silnější v modré, takže drobné částice mívají modravé halo — luma mu
  dá váhu jen 0,114, průměr 0,333. Bonus: průměrování tří nezávislých kanálů
  zmenší σ šumu až √3×. (U demozaikovaného snímku je zisk menší, protože kanály
  jsou spolu korelované.)
* **maximum kanálů** je nejcitlivější na částici, která svítí jen v jednom
  kanálu. Pozor ale na to, co udělá se šumem: maximum tří nezávislých normálních
  veličin **není** normální. Změřeno na 3 milionech vzorků se σ = 1:

  | projekce | medián | σ | MAD × 1,4826 |
  |----------|--------|---|--------------|
  | jeden kanál | +0,000 | 1,000 | 0,999 |
  | průměr 3 kanálů | −0,001 | 0,577 | 0,577 |
  | maximum 3 kanálů | **+0,818** | 0,748 | 0,741 |

  Rozptyl tedy neroste — ale rozdělení se **posune nahoru a zešikmí doprava**.
  Konstantní posun práh `medián + k·σ` sám pohltí, těžší pravý ocas ale ne, takže
  při stejné sigmě propadne víc falešných detekcí. Když sáhnete po maximu, sigmu
  radši o půl stupně zvedněte.

---

## 4. Krok 2 — geometrie: binning a výřez

```python
def apply_geometry(image: np.ndarray, binning: int, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Aplikuje binning (INTER_AREA) a ořez ROI na float32 snímek."""
    out = image
    if binning > 1:
        h, w = out.shape[:2]
        new_w = max(1, w // binning)
        new_h = max(1, h // binning)
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)
```
<sub>`analyzer.py`, funkce `apply_geometry()`</sub>

**Binning** je průměrování bloků 2×2 nebo 4×4 pixelů (`INTER_AREA` dělá přesně
to). Není to jen zrychlení — je to i **zlepšení poměru signál/šum**. Průměrem
*n* pixelů klesne směrodatná odchylka šumu √*n*-krát, zatímco signál rozlehlejší
než jeden pixel zůstane. Binning 2×2 tedy zdvojnásobí SNR difuzního zamlžení.

Cena: objekty menší než blok se rozmažou. Proto se binning volí podle rozlišení —
u 4K se ve výchozím stavu použije 2×2, u Full HD nic.

> **Pozor na jednotku plochy.** Po binningu 2×2 má jeden pixel plochu 4 pixelů
> původního snímku. Aby „minimální plocha 3 px“ znamenala pořád totéž, přepočítává
> se práh do souřadnic po binningu (`area_factor` v `classify_components()`) a
> výsledné plochy zpět (`px_scale` v `analyze_prepared_frame()`). Díky tomu vyjdou
> stejná čísla bez ohledu na zvolený binning.

---

## 5. Krok 3 — odečet referenčního pozadí

Toto je krok, který z „hezkého obrázku“ dělá **měření**.

```python
    diff = cv2.subtract(image, bias)
    reference_offset = float(level_offset)
    if abs(reference_offset) > 1e-3:
        diff = diff - np.float32(reference_offset)
```
<sub>`analyzer.py`, funkce `analyze_prepared_frame()`</sub>

Rozdíl **se neořezává na nulu**. Záporné hodnoty (pixel je tmavší než reference)
jsou fyzikálně nesmyslné jako signál — ale jsou to nejlepší data o šumu, jaká
máme, protože v nich žádná kontaminace není. Uvidíme v kroku 5.

### Odkud pozadí pochází

Jsou dvě možnosti a liší se v jedné důležité věci:

| Zdroj | Jak vznikne | Daň |
|-------|-------------|-----|
| **Prvních N snímků série** | medián (nebo průměr) prvních N snímků | těchto N snímků se **z výsledků vyřadí** — porovnávaly by se samy se sebou |
| **Reference ze složky `reference`** | pole `mean_mono` z `.npz` (průměr 16 tmavých snímků) | žádná, vyhodnotí se všechny snímky |

Proč **medián** a ne průměr? Kdyby se do jednoho z bias snímků připletla částice,
průměr ji rozmaže do reference a ta pak bude ve **všech** dalších snímcích
odečítat na jejím místě příliš — objeví se tam trvalá záporná díra. Medián
z pěti snímků jedinou výjimečnou hodnotu úplně ignoruje.

### Srovnání úrovně u externí reference

Reference vznikla dřív, takže se od snímků může lišit konstantním posunem jasu
(teplota senzoru, jas LED). Bez ošetření by se posun +6 ADU projevil jako
**100 % zamlžení na dokonale čistém sklíčku**. Posun se proto odhaduje takhle:

```python
    levels: List[float] = []
    for path in picks:
        try:
            frame = load_frame(path, full_scale=bias.full_scale, mono_mode=params.mono_mode)
        except FrameReadError:
            continue
        if frame.data.shape != bias.source_shape:
            continue
        prepared = apply_geometry(frame.data, bias.binning, params.roi)
        diff = cv2.subtract(prepared, match_shape(bias.data, prepared.shape[:2]))
        _haze_full, haze_small = separate_haze(diff)
        if haze_small.size:
            levels.append(float(np.percentile(haze_small, REFERENCE_LEVEL_PERCENTILE)))

    if not levels:
        return 0.0
    offset = min(levels)
    return offset if abs(offset) >= REFERENCE_LEVEL_DEADBAND_ADU else 0.0
```
<sub>`analyzer.py`, funkce `estimate_reference_offset()`</sub>

Klíčové je **`min(levels)` přes celou sérii**, ne hodnota z jednoho snímku.
Logika stojí na bodu 1 z kapitoly o fyzice: kontaminace jas jen přidává. Rozdíl
přístrojového původu je proto v každém snímku **stejný**, kdežto zamlžení se
v čase mění — nejnižší naměřená úroveň za celou sérii je tedy nejlepší odhad
skutečné nuly.

Kdyby se posun počítal pro každý snímek zvlášť, odečetl by se i skutečný opar —
vyzkoušeno: 14 ADU mlhy přes celé pole vyšlo jako 18 % pokrytí místo 100 %.

---

## 6. Krok 4 — rozklad na zamlžení a ostrou složku

Dechová mlha, kondenzát a zaschlý film mají jinou prostorovou charakteristiku než
prach: jsou **hladké a rozlehlé**, kdežto částice jsou **malé a ostré**. Rozklad
podle prostorové frekvence je proto přirozený způsob, jak je oddělit.

```python
    h, w = diff.shape[:2]
    small_w = max(16, w // downscale)
    small_h = max(16, h // downscale)

    small = cv2.resize(diff, (small_w, small_h), interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    small = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=2.0, sigmaY=2.0)

    haze_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return haze_full, small
```
<sub>`analyzer.py`, funkce `separate_haze()`</sub>

Čtyři operace, každá má svůj důvod:

1. **Zmenšení 16×** (`INTER_AREA`) — samo o sobě je to dolní propust a je to
   256× levnější než rozostřovat plné rozlišení.
2. **Morfologické otevření** (eroze + dilatace) — tohle je ten chytrý krok.
   Bez něj by jasná částice po zmenšení zvedla lokální průměr a analýza by kolem
   ní hlásila „opar“. Otevření odstraní všechno, co je menší než jádro, takže do
   odhadu oparu vstoupí jen skutečně rozlehlé struktury.
3. **Gaussovo rozostření** — vyhladí schody po morfologii.
4. **Zvětšení zpět** a odečtení:

```python
    sharp = cv2.subtract(diff, haze_full)
```
<sub>`analyzer.py`, funkce `analyze_prepared_frame()`</sub>

Výsledek: `diff = haze + sharp`, přičemž `haze` obsahuje jen pomalé změny a
`sharp` jen ostré detaily. Na obrázku v kapitole 2 jsou to panely 4 a 5.

Zamlžení se pak měří **prostým prahem** na `haze_small` (výchozí 4 ADU) — na
hladké složce nemá smysl počítat sofistikovaný šumový práh, protože šum už je
z ní vyhlazený.

---

## 7. Krok 5 — odhad šumu a práh

**Toto je nejcitlivější místo celé analýzy.** Práh se odvozuje ze šumu, takže
chybný odhad šumu = chybné všechno.

![Odhad šumu a práh](img/02_sum_a_prah.png)

Levý graf ukazuje, jak rozdělení jasu ostré složky doopravdy vypadá: úzký
symetrický vrchol kolem nuly (šum) a **dlouhý pravý ocas** (kontaminace).
Práh musí ležet přesně mezi nimi.

### Naivní řešení a proč nefunguje

Nabízí se změřit rozptyl celého rozdělení (`std`) nebo robustní MAD. Obojí
selže, a to dvěma opačnými způsoby:

* **`std` z celého snímku** zahrne do „šumu“ i samotnou kontaminaci. Čím
  špinavější sklíčko, tím vyšší práh — a tím míň se toho najde. Metoda je sama
  proti sobě.
* **MAD z celého rozdělení** selže na bias snímcích. U mediánového biasu je
  u snímku, který sám do biasu vstoupil, **přes polovinu pixelů diference přesně
  nulová** → MAD vyjde skoro nula → práh spadne pod šum → analýza „najde“ stovky
  až tisíce neexistujících částic. Pravý sloupec obrázku výše: **174 objektů na
  čistém snímku 700×400**, tedy asi 621 na megapixel — na 4K snímku by to bylo
  přes 5 000 fantomů.

### Použité řešení: rozdíly sousedních pixelů

```python
    sigma = 0.0
    if patch.ndim == 2 and patch.shape[1] > 1:
        deltas = (patch[:, 1:] - patch[:, :-1]).ravel()
        sigma = float(np.median(np.abs(deltas - np.median(deltas)))) * 1.4826 / math.sqrt(2.0)
```
<sub>`analyzer.py`, funkce `estimate_noise()`</sub>

Myšlenka: **šum se mění od pixelu k pixelu, scéna ne.** Rozdíl sousedních pixelů
proto skoro celý pochází ze šumu — struktura scény (i jasná částice) je spojitá
a z rozdílu se odečte.

Odvození konstant:

* `MAD × 1,4826` — pro normální rozdělení platí `MAD = 0,6745 σ`, tedy
  `σ = MAD / 0,6745 = 1,4826 × MAD`. Medián absolutních odchylek je na rozdíl od
  `std` odolný vůči odlehlým hodnotám: může jich být až 50 % a odhad se nezmění.
* `/ √2` — rozdíl dvou nezávislých veličin se stejným rozptylem má rozptyl
  dvojnásobný, tedy směrodatnou odchylku `√2 σ`. Dělením se vracíme k σ jednoho
  pixelu.

V obrázku výše dal tento odhad `σ = 0,803 ADU` proti naivním `0,495` — a **0
falešných částic** místo 174.

### Práh

```python
    sharp_median, sharp_sigma = estimate_noise(sharp)
    if params.threshold_mode == "absolute":
        applied_threshold = float(params.absolute_threshold)
    else:
        applied_threshold = sharp_median + float(params.sigma) * sharp_sigma
    applied_threshold = max(applied_threshold, float(params.min_threshold_adu), 0.5)

    mask_sharp = (sharp > applied_threshold).astype(np.uint8)
```
<sub>`analyzer.py`, funkce `analyze_prepared_frame()`</sub>

`medián + k·σ` je klasický **sigma-clipping**. Volba `k = 4` říká: falešně
pozitivní pixel připustíme s pravděpodobností asi 3·10⁻⁵. Na 8megapixelovém
snímku to je ~260 osamocených pixelů — a ty pak odfiltruje podmínka minimální
plochy 3 px, protože pravděpodobnost, že tři takové sousedí, je zanedbatelná.
Ta dvojice podmínek (jas **a** plocha) je důvod, proč analýza na čistém snímku
skutečně vrací nulu.

`min_threshold_adu` (výchozí 1,5) je pojistka pro případ dokonale hladkého
pozadí, kde by σ vyšlo nesmyslně malé.

---

## 8. Krok 6 — segmentace na objekty

```python
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_sharp, connectivity=8, ltype=cv2.CV_32S
    )
```
<sub>`analyzer.py`, funkce `analyze_prepared_frame()`</sub>

Binární maska se rozpadne na **souvislé oblasti**. `connectivity=8` znamená, že
za sousedy se považují i pixely přes roh — u tenkého šikmého vlákna je to nutné,
jinak by se rozpadlo na řetízek samostatných bodů.

`stats` je matice, kde každý řádek popisuje jeden objekt: `LEFT`, `TOP`, `WIDTH`,
`HEIGHT`, `AREA`. Řádek 0 je vždy pozadí.

> **`ltype=cv2.CV_32S` tady není detail.** Původní verze aplikace používala
> `CV_16U`, což omezuje počet objektů na 65 535. U zašuměného 4K snímku se ten
> limit překročí snadno a OpenCV vyhodí výjimku *„Total number of labels
> overflowed label type“* — což byla jedna ze tří příčin, proč aplikace padala.

---

## 9. Krok 7 — klasifikace typu kontaminace

Máme objekty, teď jim potřebujeme přiřadit typ. Rozhodují **dvě čísla**: plocha
a protáhlost.

### Jak se měří protáhlost

Nejjednodušší nápad — poměr stran opsaného obdélníku — **vážně nefunguje**:

![Klasifikace tvaru](img/03_tvary.png)

Vlákno pod 45° má opsaný obdélník čtvercový, tedy poměr **1,00**. Podle něj by
šlo o kulatý shluk. Rotací objektu by se změnila jeho klasifikace, což je pro
měřicí metodu nepřijatelné.

Řešení jsou **momenty druhého řádu**. Objekt se nahradí elipsou, která má stejné
rozložení hmoty, a protáhlost se počítá z jejích poloos:

```python
    counts = np.bincount(lab, minlength=n).astype(np.float64)
    counts_safe = np.maximum(counts, 1.0)
    sx = np.bincount(lab, weights=xs, minlength=n)
    sy = np.bincount(lab, weights=ys, minlength=n)
    sxx = np.bincount(lab, weights=xs * xs, minlength=n)
    syy = np.bincount(lab, weights=ys * ys, minlength=n)
    sxy = np.bincount(lab, weights=xs * ys, minlength=n)

    mx = sx / counts_safe
    my = sy / counts_safe
    # +1/12 = rozptyl rovnoměrného rozdělení uvnitř pixelu (diskrétní korekce)
    cxx = sxx / counts_safe - mx * mx + 1.0 / 12.0
    cyy = syy / counts_safe - my * my + 1.0 / 12.0
    cxy = sxy / counts_safe - mx * my

    tmp = np.sqrt(np.maximum(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy))
    lam_major = 0.5 * (cxx + cyy + tmp)
    lam_minor = np.maximum(0.0, 0.5 * (cxx + cyy - tmp))

    major_axis = 4.0 * np.sqrt(np.maximum(lam_major, 0.0))
    minor_axis = 4.0 * np.sqrt(lam_minor)
    elongation = major_axis / np.maximum(minor_axis, 1.0)
```
<sub>`analyzer.py`, funkce `_shape_descriptors()`</sub>

Co se tu děje matematicky:

1. `cxx`, `cyy`, `cxy` je **kovarianční matice** souřadnic pixelů objektu.
2. `lam_major` a `lam_minor` jsou její **vlastní čísla** — vzorec je uzavřené
   řešení pro matici 2×2. Odpovídají rozptylu podél hlavní a vedlejší osy.
3. `4·√λ` dává délku osy ekvivalentní elipsy (pro rovnoměrně vyplněnou elipsu
   platí `λ = (a/2)²/4`, odkud `a = 4√λ`).
4. Vlastní čísla jsou **invariantní vůči otočení**, takže vlákno pod 45° vyjde
   stejně jako vodorovné — v obrázku výše 29,6 vs. 21,4 místo 1,00 vs. 21,8.

Detail `+ 1/12`: pixel není bod, ale čtvereček. Rozptyl rovnoměrného rozdělení na
intervalu délky 1 je 1/12. Bez téhle korekce by jednopixelové objekty vyšly
s nulovým rozptylem a protáhlostí dělenou nulou.

Za povšimnutí stojí, že celý výpočet je **vektorizovaný přes `np.bincount`** —
žádná smyčka přes objekty. Momenty všech 50 000 objektů se spočítají jedním
průchodem polem, jinak by 4K snímek trval sekundy místo milisekund.

### Vlastní rozhodovací pravidlo

```python
    valid = areas >= min_area
    valid[0] = False  # pozadí

    is_fiber = valid & (elongation >= aspect_limit) & (major_axis >= min_fiber_len)
    is_cluster = valid & ~is_fiber & (areas >= cluster_min)
    is_point = valid & ~is_fiber & ~is_cluster
```
<sub>`analyzer.py`, funkce `classify_components()`</sub>

Rozhodovací strom je záměrně plochý a v tomto pořadí:

1. **Moc malé zahoď** (šum, výchozí < 3 px).
2. **Vlákno** = protáhlé **a zároveň** dost dlouhé. Obě podmínky jsou nutné.
   Třípixelová čárka ze šumu má protáhlost přesně 3,0 — tedy nad výchozím
   limitem 2,8 — a bez podmínky na délku by prošla jako vlákno. Délka její
   hlavní osy je ale 3,5 px, takže ji zamítne práh 12 px:

   | čárka | protáhlost | délka hlavní osy |
   |-------|-----------|------------------|
   | 2 px | 2,00 | 2,3 px |
   | 3 px | 3,00 | 3,5 px |
   | 4 px | 4,00 | 4,6 px |
   | 6 px | 6,00 | 6,9 px |
3. **Velký shluk** = není vlákno a plocha ≥ 100 px (kapka, aglomerát).
4. **Mikročástice** = všechno ostatní (běžné smítko prachu).

Pořadí není libovolné: vlákno se testuje **první**, protože dlouhé vlákno může
mít snadno přes 100 px plochy a jinak by spadlo do shluků.

Kategorie se zapíšou do **lookup tabulky** indexované číslem objektu, takže
obarvení celého snímku je jediná indexovací operace:

```python
    classified = particles.category_lut[labels]        # uint8, 0–3
    mask_particles = classified > 0
```
<sub>`analyzer.py`, funkce `analyze_prepared_frame()`</sub>

---

## 10. Krok 8 — z masek na čísla

### Disjunktní rozklad pokrytí

Masky se **překrývají** — částice běžně leží uvnitř zamlžené oblasti. Kdyby se
plochy prostě sečetly, součet by přesáhl celkové pokrytí a graf složení by lhal.
Proto:

```python
    particle_area_binned = (
        particles.point_area_px + particles.cluster_area_px + particles.fiber_area_px
    )
    haze_only_px = max(0, total_area_px_binned - particle_area_binned)
    haze_only_coverage_pct = 100.0 * haze_only_px / total_pixels if total_pixels else 0.0
```
<sub>`analyzer.py`, funkce `analyze_prepared_frame()`</sub>

Díky tomu platí přesně:
`zamlžení bez částic + mikročástice + shluky + vlákna = celkové pokrytí`,
a vrstvený graf v záložce 🧩 dává smysl.

### Nehomogenita a těžiště

```python
    grid = cv2.resize(
        (mask_u8 > 0).astype(np.float32), (zones, zones), interpolation=cv2.INTER_AREA
    )
    mean_zone = float(grid.mean())
    if mean_zone <= 1e-9:
        return 0.0, cx_pct, cy_pct

    cv_score = float(grid.std()) / mean_zone
    max_cv = math.sqrt(zones * zones - 1)
    heterogeneity = 100.0 * min(1.0, cv_score / max_cv)
```
<sub>`analyzer.py`, funkce `spatial_distribution()`</sub>

Snímek se rozdělí na mřížku 8×8 a spočítá se **variační koeficient** (σ/μ)
pokrytí mezi zónami. Normuje se svým teoretickým maximem: kdyby všechna
kontaminace byla v jediné z *N* zón, vyjde `CV = √(N−1)`. Hodnota je tak
opravdu v rozsahu 0–100 %, kde 0 % = rovnoměrné znečištění a 100 % = všechno na
jedné hromádce. Rozdíl mezi „rovnoměrný prach ze vzduchu“ a „někdo se dotkl
rohu“ je právě tohle číslo.

### Skóre čistoty

```python
    p_cov = 45.0 * (1.0 - math.exp(-max(0.0, coverage_pct) / 2.0))
    p_haze = 30.0 * (1.0 - math.exp(-max(0.0, haze_mean_adu) / 8.0))
    p_part = 25.0 * (1.0 - math.exp(-max(0.0, particle_density_per_mpx) / 120.0))
    return float(max(0.0, min(100.0, round(100.0 - (p_cov + p_haze + p_part), 1))))
```
<sub>`analyzer.py`, funkce `compute_cleanliness_score()`</sub>

Je to **pomocné, nikoliv fyzikální číslo** — jeden index pro rychlé srovnání
snímků. Rozpočet penalizací je 45 + 30 + 25 bodů. Saturující exponenciála
`1 − e^(−x/x₀)` je zvolená proto, že:

* je spojitá a monotónní (skóre nikdy neskočí),
* na malých hodnotách je skoro lineární (rozliší 0,1 % od 0,3 % pokrytí),
* nahoře saturuje (rozdíl mezi 50 % a 80 % pokrytí už je jedno — obojí je zle).

Hustota částic se počítá **na megapixel**, aby skóre nezáviselo na rozlišení
kamery ani na zvoleném binningu.

### Ostrost jako kontrola kvality

```python
    lap = cv2.Laplacian(small, cv2.CV_32F, ksize=3)
    return float(lap.var())
```
<sub>`analyzer.py`, funkce `focus_score()`</sub>

Variance Laplaciánu je standardní míra ostrosti: Laplacián je druhá derivace,
takže reaguje na hrany, a ostrý obraz jich má víc. **Neměří kontaminaci** —
slouží ke kontrole, že snímek nerozmazaly vibrace nebo ujetý autofokus.
Když v sérii najednou spadne, výsledkům z toho místa nevěřte.

---

## 11. Krok 9 — čas: derivace a fáze děje

![Časová řada a fáze](img/04_faze.png)

### Derivace

Nabízelo by se počítat `(x[i+1] − x[i]) / (t[i+1] − t[i])`. U rychlé série to je
špatně: `dt` je malé, šum se dělí malým číslem a derivace je nepoužitelná.
Používá se proto **lokální lineární regrese** (Savitzky–Golay 1. řádu) přes okno
5 snímků:

```python
    half = max(1, window // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        t = times[lo:hi]
        v = values[lo:hi]
        t_mean = t.mean()
        denom = float(((t - t_mean) ** 2).sum())
        if denom <= 1e-12:
            slopes[i] = 0.0
        else:
            slopes[i] = float(((t - t_mean) * (v - v.mean())).sum() / denom)
```
<sub>`analyzer.py`, funkce `_local_slope()`</sub>

To je vzorec pro směrnici metodou nejmenších čtverců — proloží se okolím přímka
a vezme se její sklon. Šum se zprůměruje, skutečný trend zůstane.

### Klasifikace fáze

```python
    if abs(d_cov) > cov_thresh:
        return "nárůst / zamlžování" if d_cov > 0 else "odpařování / ústup"
    if abs(d_haze) > haze_thresh:
        return "nárůst / zamlžování" if d_haze > 0 else "odpařování / ústup"
    return "stabilní"
```
<sub>`analyzer.py`, funkce `_classify_phase()`</sub>

Dvě věci, které vypadají triviálně, ale jsou obě opravou konkrétní chyby:

1. **Znaménko určuje vždy jen jedna veličina.** Původní podmínka
   `if d_pokrytí > práh or d_jas > práh` označila **klesající** pokrytí za nárůst,
   pokud zároveň rostl průměrný jas.
2. **Doplňkové kritérium je zamlžení, ne průměrný jas.** Průměrný jas se počítá
   jen přes kontaminované pixely — když opar zmizí a zůstanou jen jasné částice,
   **mechanicky vyskočí nahoru**, i když kontaminace ubývá. Je to ukazatel, který
   se v tomhle okamžiku chová přesně opačně, než by člověk čekal.

### Adaptivní práh „už je to změna“

```python
    peak = float(np.percentile(np.abs(rates), 95))
    return max(floor, fraction * peak)
```
<sub>`analyzer.py`, funkce `_phase_threshold()`</sub>

Práh se odvozuje z dynamiky **konkrétního měření** (5 % z 95. percentilu
rychlostí), nikdy neklesne pod pevnou mez. Pevný práh by u pomalého usazování
prachu neoznačil nic a u prudkého zapaření dechem naopak úplně všechno.

---

## 12. Co která páčka dělá

| Parametr | Zvýšení znamená | Kdy sáhnout |
|----------|-----------------|-------------|
| **Sigma** (4,0) | vyšší práh → míň nalezených částic | tisíce „částic“ na čistém sklíčku → 5–6 |
| **Absolutní práh** | pevný práh v ADU místo σ | když chcete srovnatelnost mezi měřeními za cenu citlivosti |
| **Práh zamlžení** (4,0) | míň plochy označené jako opar | jemný závoj se nezachytí → 2–3; kolísá osvětlení → 5–7 |
| **Min. plocha částice** (3 px) | přísnější filtr osamocených pixelů | zbytkové falešné detekce → 5 |
| **Velký shluk od** (100 px) | posouvá hranici mikročástice ↔ shluk | podle měřítka objektivu; ověřte v prohlížeči |
| **Protáhlost vlákna** (2,8) | méně objektů projde jako vlákno | shluky se hlásí jako vlákna → 3,5 |
| **Min. délka vlákna** (12 px) | krátké šmouhy se neberou jako vlákno | šum tvoří „vlákénka“ → zvyšte |
| **Bias snímků** (3) | tišší reference, ale méně snímků ve výsledku | u externí reference se neuplatní |
| **Binning** | rychlost a SNR nahoru, detail dolů | 4K → 2×2; hledáte nejjemnější prach → plné rozlišení |
| **Měřítko µm/px** | přepočet na fyzikální jednotky | podle kalibrace objektivu; 0 = jen pixely |

Nejrychlejší způsob, jak zjistit, jestli je nastavení dobré: **záložka 🖼️ Snímky**.
Přepněte na „barevná klasifikace“ a porovnejte s originálem. Když barvy sedí na
to, co vidíte očima, sedí i čísla.

---

## 13. Limity metody — co to neumí

Poctivý výčet toho, kde metoda přestává platit:

* **Dotýkající se objekty splynou v jeden.** Segmentace spojitých komponent
  nemá jak poznat, že tři slepené kuličky jsou tři. Dvě částice u sebe se
  vyhodnotí jako jeden shluk. (Šlo by řešit watershedem, ale za cenu spousty
  falešných dělení u vláken.)
* **Klasifikace je geometrická, ne materiálová.** „Vlákno“ znamená *protáhlý
  objekt*, ne *textilní vlákno*. Škrábanec na sklíčku je pro analýzu totéž.
* **Jas ani plocha neříkají, jak je částice velká.** Účinnost rozptylu závisí na
  poměru velikosti a vlnové délky velmi silně a nemonotónně: u částic mnohem
  menších než vlnová délka roste jako *d*⁶ (Rayleigh), u větších přechází do
  složitého Mieova režimu se závislostí na materiálu i tvaru. K tomu se každý
  bodový zdroj rozmaže na difrakční skvrnu velkou jako rozlišovací mez optiky.
  Plocha nad prahem je proto **plocha rozptylového obrazu**, ne fyzický rozměr
  částice — čísla v µm² jsou dobrá na srovnávání mezi snímky téže série,
  ne jako absolutní rozměr.
* **Analýza měří změnu, ne absolutní špínu.** Cokoli, co je na sklíčku po celou
  dobu stejné, je součástí reference a nezapočítá se. S biasem ze série to platí
  z definice (referencí je začátek měření). U externí reference to platí navíc
  o plošné složce: srovnání úrovně odečte konstantní posun, a je-li ten posun
  ve skutečnosti rovnoměrná špína, zmizí s ním. (Pokud je posun menší než
  0,5 ADU, korekce se vůbec nezapne — u reference pořízené těsně před měřením
  se tedy prakticky neuplatní.)
* **Plošné zamlžení přes 100 % pole nelze odlišit od posunu jasu.** Když je
  celé pole rovnoměrně zamlžené, je to matematicky totéž jako jinak nastavená
  expozice. Pomůže jen reference pořízená těsně předtím.
* **Metoda nepozná, co se změnilo mezi referencí a měřením** kromě sklíčka.
  Když se mezitím sáhne na ostření nebo na osvětlení, výsledky nejsou srovnatelné.
  Proto se hlásí odstup reference nad 6 hodin.

---

## 14. Jak si to osahat

Analytické jádro je bez závislosti na GUI, takže se dá volat přímo. Tenhle skript
projde jeden snímek a vypíše, co se v každém kroku stalo:

```python
import numpy as np
from datetime import datetime

from analyzer import AnalysisParams, analyze_prepared_frame, estimate_noise, separate_haze
from frameio import load_frame

params = AnalysisParams(binning=1, sigma=4.0, haze_threshold=4.0)

bias = load_frame(r"C:\...\reference_snimek.png").data
frame = load_frame(r"C:\...\df_00042_20260908_143500_000.png").data

diff = frame - bias
haze, haze_small = separate_haze(diff)
sharp = diff - haze
median, sigma = estimate_noise(sharp)

print(f"šum pozadí:      σ = {sigma:.3f} ADU")
print(f"práh:            {median + 4 * sigma:.2f} ADU")
print(f"pixelů nad prahem: {(sharp > median + 4 * sigma).sum()}")

t0 = datetime.now()
metrics, masks = analyze_prepared_frame(
    image=frame, bias=bias, params=params, index=0, filename="test.png",
    filepath="test.png", timestamp=t0, t0=t0, binning=1, generate_masks=True)

print(f"pokrytí:   {metrics.total_coverage_pct:.3f} %")
print(f"částice:   {metrics.point_count} mikro / {metrics.cluster_count} shluků "
      f"/ {metrics.fiber_count} vláken")
print(f"zamlžení:  {metrics.haze_coverage_pct:.2f} % plochy")
print(f"čistota:   {metrics.cleanliness_score:.1f} / 100")
print("masky k dispozici:", list(masks))
```

S `generate_masks=True` dostanete slovník s mezivýsledky (`diff`, `haze`,
`sharp`, `mask_haze`, `mask_points`, `mask_clusters`, `mask_fibers`,
`mask_total`) — přesně z nich vznikly obrázky v tomto dokumentu.

### Obrázky v tomto dokumentu

Nejsou to kresby — vyrábí je skript, který volá skutečné jádro. Můžete si je
přegenerovat proti vlastní referenci a uvidíte svoje pozadí:

```powershell
py tools\make_docs_figures.py --reference "C:\Users\Programovani\Downloads\BMS fotky\reference\reference_20260908_142910.npz"
```

Bez vlastních dat si sérii vygenerujete:

```powershell
py tools\make_demo_series.py --out "%USERPROFILE%\Downloads\demo_darkfield" --frames 40
```

A když chcete vidět, jak se která změna projeví na číslech, testy jsou docela
dobrá čítanka — každý test má v docstringu, co a proč ověřuje:

```powershell
py -m pytest -q
py -m pytest tests\test_analyzer.py -q -k noise -v     # odhad šumu
py -m pytest tests\test_analyzer.py -q -k classif -v   # klasifikace tvarů
py -m pytest tests\test_reference.py -q -k level -v    # srovnání úrovně reference
```

---

## Kam dál v kódu

| Chci rozumět… | Otevřete |
|---------------|----------|
| načítání, bitové hloubce, barvám | `frameio.py` |
| výběru reference podle času | `reference.py` |
| celé analýze jednoho snímku | `analyzer.py`, `analyze_prepared_frame()` |
| paralelnímu zpracování série | `analyzer.py`, `analyze_series()` |
| grafům a exportu | `exporter.py` |
| vizuální kontrole s maskami | `viewer.py` |
