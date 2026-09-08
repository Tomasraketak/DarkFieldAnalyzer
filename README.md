# Dark-Field Contamination Analyzer

Automatické vyhodnocení časových řad snímků z **mikroskopie v temném poli**
(dark-field) pořízených černobílou kamerou. Aplikace měří, kolik kontaminace
(prach, vlákna, kapky, kondenzát) přibylo na witness sklíčku, jak rychle a v
jakých fázích.

Cíl: Windows 11, Dell Vostro 7500 (Intel Core i7, 16 GB RAM), Python 3.10+.

---

## Rychlý start

```powershell
cd C:\cesta\k\DarkFieldAnalyzer
instalovat_knihovny.bat      # jednorázově
spustit_analyzu.bat          # spustí grafické rozhraní
```

Bez reálných dat si můžete vygenerovat ukázkovou sérii:

```powershell
py tools\make_demo_series.py --out "%USERPROFILE%\Downloads\demo_darkfield" --frames 40
```

---

## Co bylo opraveno oproti předchozí verzi

Aplikace padala hned po stisku tlačítka **SPUSTIT ANALÝZU**. Příčiny byly tři
a všechny jsou pokryté regresními testy:

| # | Příčina | Projev | Oprava |
|---|---------|--------|--------|
| 1 | `gui.py` se připojoval na neexistující metodu `self.on_worker_error` | `AttributeError` uvnitř Qt slotu; PyQt6 v takovém případě volá `qFatal` a **ukončí celý proces** – okno zmizelo bez hlášky | slot doplněn, přidán `install_exception_hook()` a pracovní vlákno posílá celý traceback |
| 2 | `cv2.connectedComponentsWithStats(..., ltype=cv2.CV_16U)` | výjimka `Total number of labels overflowed label type`, jakmile snímek obsahoval přes 65 535 objektů (u zašuměného 4K snímku běžné) | typ labelů `CV_32S` |
| 3 | Prohlížeč počítal náhled ve 4× zmenšení, ale bias měl v 2× zmenšení | `cv2.subtract` vyhodil výjimku hned po dokončení analýzy | náhled se počítá v geometrii analýzy + pojistka `match_shape()` |

Další opravené problémy:

- **Cesty s diakritikou.** `cv2.imread()` na Windows vrátí `None` u každé cesty
  jako `C:\Users\Jiří\Měření` – snímek se tiše přeskočil, analýza skončila s nula
  snímky a GUI zůstalo viset s vypnutými tlačítky. Nyní se čte přes
  `np.fromfile` + `cv2.imdecode` a nenačtený soubor se nahlásí.
- **Rozbitý odhad šumu.** Šum se odhadoval z už oříznuté (saturované) uint8
  diference, kde je polovina rozdělení uříznutá. Práh proto vycházel nahodile –
  na některých snímcích se „našly“ desítky tisíc neexistujících částic.
- **Fáze děje.** Podmínka `if d_pokrytí > práh or d_jas > práh` označila klesající
  pokrytí za nárůst, když zároveň rostl průměrný jas.
- **Dělení nulou** při prázdném výsledku, tuhnoucí GUI při tisících snímků,
  chybějící ošetření chyb při exportu do otevřeného souboru v Excelu.

---

## Co je v analýze nového

### Přesnost měření

- **Celý řetězec ve float32** místo saturované uint8 aritmetiky. Rozdíl
  `snímek − bias` si zachová i zápornou část, ze které jde jako z jediné
  poctivě odhadnout šum pozadí.
- **Odhad šumu z rozdílů sousedních pixelů** (vysokofrekvenční odhad). Měří
  skutečný šum senzoru, ne strukturu scény – na snímku, kde je půlka plochy
  pokrytá zaschlým filmem, by klasický odhad z celkového rozdělení tuto
  strukturu započítal jako šum a jemné částice by pod prahem zmizely.
- **Práh bez zaokrouhlení na celé ADU.** Původní `int(round(práh))` zahazoval
  až 0,5 ADU, což je u prahu 3 ADU šestinová chyba.
- **Binning průměrováním** (INTER_AREA) místo podvzorkování `img[::2, ::2]`.
  Podvzorkování zahodí tři čtvrtiny pixelů a s nimi náhodně i část částic;
  průměrování naopak zlepšuje poměr signál/šum.
- **Mediánový master bias.** Jediná náhodná částice v referenčním snímku už
  nezkazí měření celé série.
- **Podpora 10/12/14/16bitových kamer.** Data se převádějí na jednotnou škálu
  0–255 ADU, takže všechny prahy mají stále stejný význam.
- **Jednotná definice pokrytí.** Dřív se počítala jinak při zobrazení a jinak
  při analýze, takže se čísla v tabulce a v prohlížeči neshodovala.

### Klasifikace objektů

- **Protáhlost z momentů druhého řádu** (poměr os ekvivalentní elipsy) místo
  poměru stran opsaného obdélníku. Šikmé vlákno pod 45° má opsaný obdélník
  téměř čtvercový – dřív se proto klasifikovalo jako shluk.
- **Plně vektorizovaná klasifikace.** Původní smyčka v Pythonu přes všechny
  objekty byla u zašuměných snímků s desítkami tisíc objektů úzkým hrdlem.

### Nové sledované veličiny

| Veličina | K čemu je |
|----------|-----------|
| Hustota částic [ks/Mpx] | srovnatelná napříč rozlišeními a binningem |
| Medián / P90 / maximum plochy částice | rozdělení velikostí, ne jen počet |
| Ekvivalentní průměr částice [µm] | fyzikální velikost při zadané kalibraci |
| Celková délka vláken [px] | míra „vlákitosti“ znečištění |
| Ostrost (variance Laplaciánu) | odhalí rozostřený snímek nebo otřes stativu |
| Pokrytí zamlžením bez částic [%] | disjunktní rozklad pokrytí pro graf složení |
| Úroveň pozadí a SNR | kontrola stability osvětlení a expozice |
| Šum pozadí σ a použitý práh | doložitelnost detekční meze v protokolu |

- **Skóre čistoty** je nově složené ze tří saturujících penalizací (pokrytí 45 b.,
  opar 30 b., hustota částic 25 b.). Původní lineární vzorec padal na nulu už
  při 8 % pokrytí a jeho hodnota závisela na rozlišení kamery.
- **Nehomogenita** se počítá v mřížce 8×8 a je normovaná svým teoretickým
  maximem, takže hodnota je skutečně v rozsahu 0–100 %.
- **Rychlosti změn** se počítají lokální lineární regresí přes 5 snímků
  (Savitzky–Golay 1. řádu) místo rozdílu sousedních snímků, který u rychlých
  sérií jen zesiloval šum. Prahy pro fáze se odvozují z dynamiky měření.

### Provoz

- **Paralelní zpracování** s ohledem na paměť: počet vláken se automaticky sníží,
  aby analýza notebook nevytlačila do odkládacího souboru.
- **Chyba u jednoho souboru sérii nezastaví** – skončí v seznamu chyb.
- **Dávkový režim** `--batch` zpracuje všechny podsložky jedním příkazem.
- **Výřez (ROI)** v GUI i na příkazové řádce.
- **Nastavení se pamatuje** mezi spuštěními (Qt QSettings).
- **Export** navíc ukládá strojově čitelný souhrn v JSON; CSV má BOM a
  desetinnou čárku, takže se korektně otevře v českém Excelu.

Naměřený výkon (4K snímky, 12 kusů, testovací stroj):

| Režim | Čas na snímek | Špička RAM |
|-------|---------------|------------|
| binning 2×2, 1 vlákno | ~250 ms | ~200 MB |
| binning 2×2, 4 vlákna | ~150 ms | ~460 MB |
| plné 4K, auto vlákna | ~370 ms | ~900 MB |

---

## Grafické rozhraní

**Levý panel**

1. **Složka s měřeními** – výchozí cesta je
   `C:\Users\Programovani\Downloads\BMS fotky`; tlačítkem *Procházet…* vyberete jinou
   (naposledy zvolená se pamatuje).
   V seznamu se objeví všechny podsložky se snímky **i samotná zvolená složka**,
   pokud snímky obsahuje přímo (dřív se nezobrazila a vypadalo to, že tam nic není).
2. **Parametry analýzy** – rozlišení/binning, počet a metoda bias snímků, režim a
   hodnota prahu, práh zamlžení, hranice hotspotu, minimální plocha částice,
   hranice velkého shluku, protáhlost a délka vlákna, kalibrace µm/px, počet
   vláken CPU, ROI. Každé pole má nápovědu po najetí myší.
3. **Spuštění a export** – průběh, zastavení, export CSV + grafy + JSON.

**Pravý panel**

- 📈 Pokrytí a čistota · 🔬 Typy kontaminace · 💡 Signál a hotspoty ·
  ⚡ Rychlost a nehomogenita
- 🧩 **Složení kontaminace** – vrstvený graf, kolik procent plochy snímku zabírá
  který typ (shluky, mikročástice, vlákna, zamlžení), druhý panel totéž bez
  zamlžení ve vlastním měřítku a dole pruh s průměrným zastoupením typů.
  Vrstvy jsou disjunktní, takže jejich součet je přesně celkové pokrytí.
- 🖼️ **Vizuální kontrola** – posuvník přes snímky, šest režimů zobrazení
  (barevná klasifikace, originál, diference, opar, ostrá složka, binární maska),
  uložení náhledu do PNG.
  Barvy: <span>azurová = zamlžení, žlutá = mikročástice, červená = shluky,
  zelená = vlákna, fialová = hotspoty, žlutý kříž = těžiště kontaminace</span>.
- 📋 Datová tabulka · 🧾 Souhrn měření · 📖 Průvodce s popisem všech veličin

---

## Příkazová řádka

```powershell
# jedna složka
py main.py --folder "C:\...\darkfield_20260907_145621" --bias 3 --binning 2 --scale 0.35

# všechny podsložky najednou
py main.py --folder "C:\...\BMS fotky" --batch

# jen výřez, absolutní práh, bez výpisu průběhu
py main.py --folder "C:\...\mereni" --roi 200,100,1500,900 --mode absolute --absolute 10 --quiet
```

Nejdůležitější přepínače (`py main.py --help` vypíše všechny):

| Přepínač | Význam | Výchozí |
|----------|--------|---------|
| `--bias`, `--bias-method` | počet a metoda bias snímků | 3, `median` |
| `--binning` | 0 = auto, jinak 1 / 2 / 4 | 0 |
| `--mode`, `--sigma`, `--absolute` | režim a hodnota prahu | `sigma`, 4.0, 12.0 |
| `--haze` | práh difuzního zamlžení [ADU] | 4.0 |
| `--min-area`, `--cluster-area` | plošné hranice částic [px] | 3, 100 |
| `--aspect-ratio`, `--fiber-length` | tvarové hranice vlákna | 2.8, 12 |
| `--scale` | kalibrace [µm/px] | 1.0 |
| `--roi` | výřez `x,y,šířka,výška` | celý snímek |
| `--workers` | počet vláken (0 = auto) | 0 |
| `--include-bias` | ponechat bias snímky ve výsledné řadě | vypnuto |

---

## Metodika

### Zpracování jednoho snímku

1. Načtení a převod na float32 v jednotné škále 0–255 ADU.
2. Binning (průměrování 2×2 nebo 4×4) a případný ořez ROI.
3. `diference = snímek − bias` (včetně záporné části).
4. Odhad **oparu**: zmenšení 16×, morfologické otevření (potlačí bodové částice,
   aby nezvyšovaly odhad pozadí), Gaussovo rozostření, zpět na plné rozlišení.
5. `ostrá složka = diference − opar`.
6. Práh `medián + sigma × σ`, nejméně `min_threshold_adu` (1,5 ADU).
7. Segmentace `connectedComponentsWithStats` (CV_32S) a klasifikace:
   - **vlákno**, pokud protáhlost ≥ 2,8 a délka hlavní osy ≥ 12 px,
   - **velký shluk**, pokud plocha ≥ 100 px,
   - jinak **mikročástice**.
8. `celkové pokrytí = maska oparu ∪ maska částic`.

Plochy v pixelech se vždy přepočítávají na **pixely plného rozlišení**, takže
čísla nezávisí na zvoleném binningu.

### Proč se bias snímky vynechávají z výsledné řady

Snímek, který sám vstoupil do výpočtu biasu, se porovnává sám se sebou. Rozdíl
je u něj z principu degenerovaný (u mediánového biasu je přes polovinu pixelů
přesně nulová) a hodnoty nejsou srovnatelné se zbytkem série. Ve výchozím
nastavení se proto do výsledků nezahrnují; časová osa ale začíná u prvního
pořízeného snímku. Přepínačem `--include-bias` (resp. odškrtnutím v GUI) je
lze zobrazit – v CSV jsou označené ve sloupci *Bias snímek*.

---

## Struktura projektu

```
analyzer.py              analytické jádro (bez závislosti na GUI)
frameio.py               načítání snímků, Unicode cesty, časová razítka
exporter.py              CSV, souhrnné grafy, JSON souhrn
gui.py                   grafické rozhraní (PyQt6)
viewer.py                prohlížeč snímků s klasifikační maskou
help_text.py             text nápovědy zobrazený v GUI
main.py                  spuštění GUI i dávkové analýzy z příkazové řádky
tools/make_demo_series.py generátor ukázkové série
tests/                   automatické testy (pytest)
```

## Testy

```powershell
py -m pip install pytest
py -m pytest -q
```

37 testů pokrývá jádro, export i grafické rozhraní (běží bez obrazovky přes
`QT_QPA_PLATFORM=offscreen`), včetně regresí na všechny tři výše popsané pády.
