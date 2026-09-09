# Dark-Field Contamination Analyzer

Automatické vyhodnocení časových řad snímků z **mikroskopie v temném poli**
(dark-field), černobílých i barevných. Aplikace měří, kolik kontaminace
(prach, vlákna, kapky, kondenzát) přibylo na witness sklíčku, jak rychle a v
jakých fázích. Referenční pozadí umí převzít z reference uložené programem
BMS Cam Control (`reference/*.npz`).

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

> 📖 **Chcete rozumět tomu, jak analýza doopravdy pracuje?**
> [**docs/JAK_TO_FUNGUJE.md**](docs/JAK_TO_FUNGUJE.md) je podrobný výklad celého
> řetězce od načtení fotky po klasifikaci kontaminace — s kusy skutečného kódu,
> odvozením použitých vzorců, obrázky jednotlivých kroků a kapitolou o tom, kde
> má metoda hranice.

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

### Referenční pozadí ze složky `reference`

Záznamový program **BMS Cam Control** ukládá referenci mimo složku s měřením –
do společné složky `…\BMS fotky\reference` – jako dvojici souborů:

| Soubor | Obsah |
|--------|-------|
| `reference_20260908_142910.npz` | `mean_mono` = průměr N tmavých snímků (float32), `frames_mono`, `created_mono` |
| `reference_20260908_142910.json` | nastavení kamery, osvětlení a prahů; jen popis, analýza z něj nic nepřebírá |

- **Automatické hledání složky.** Podsložka `reference` se hledá nejdřív přímo
  u měření, pak v nadřazených složkách. Prázdná složka `reference` hledání
  nezastaví. V GUI ji lze určit i ručně.
- **Vybere se poslední reference pořízená *před* začátkem měření.** Pozdější
  nikdy nevyhraje, ani když je časově blíž – popisovala by pozadí, které v době
  měření ještě neplatilo. Existují-li jen pozdější, použije se nejbližší z nich
  a analýza to hlásí jako upozornění. Odstup nad 6 hodin se také hlásí.
- **Čas se čte z názvu souboru**, takže výběr nemusí rozbalovat obrazová data –
  ve složce s desítkami referencí by to trvalo déle než celá analýza.
- **Žádný snímek série se nespotřebuje na bias.** Vyhodnotí se všechny snímky ve
  složce, ne až od (N+1)-ního.
- **Srovnání úrovně.** Reference vznikla dřív, takže se od snímků může lišit
  konstantním posunem jasu (teplota senzoru, jas zdroje světla). Bez srovnání by
  se takový posun projevil jako plošné zamlžení přes celý snímek. Posun se
  odhaduje **jednou pro celou sérii** jako nejnižší naměřená úroveň pozadí –
  v temném poli kontaminace jas jen přidává, takže nejtmavší snímek je nejlepší
  odhad skutečné nuly. Kdyby se počítal pro každý snímek zvlášť, odečetlo by se
  i skutečné zamlžení, které má analýza naopak měřit. Posun pod 0,5 ADU se
  ignoruje; naměřená hodnota je v souhrnu i v exportovaném JSON.
- **Režimy** (GUI i `--reference`): `auto` (reference, jinak bias ze série),
  `reference` (bez reference skončí chybou), `serie` (referenci ignoruje).

Ověřeno na skutečné referenci ze 4K kamery: při referenci posunuté o −6 ADU
hlásila analýza bez srovnání 100 % zamlžení na čistém sklíčku, se srovnáním
0 % – stejně jako s referencí, která sedí.

### Zarovnání driftu sklíčka

Během dlouhého měření se sklíčko nebo kamera posune o jednotky až desítky
pixelů. Statické částice se pak přestanou krýt s referencí, zůstanou po nich
světlé půlměsíce a analýza je počítá jako novou kontaminaci.

Před analýzou se proto ze snímků sestaví „souhvězdí“ **100 nejjasnějších
prachových částic** a sleduje se, jak se během měření posouvá:

- **Hlasování o posunu.** Pro každou dvojici (částice v kotvě, částice ve
  snímku) se hlasuje pro jejich rozdíl; skutečný posun dostane tolik hlasů,
  kolik je společných částic. Zvládne to i posun větší než rozestup částic
  a nevadí mu, že kontaminace během měření přibývá. Naměřená přesnost je
  **lepší než 0,1 px**.
- **Pojistka proti vadným pixelům.** Horké pixely se s driftem nepohybují, takže
  by hlasovaly pro nulový posun. Poznají se podle toho, že jejich sousedé jsou
  na úrovni pozadí (skutečná částice je rozmazaná optikou), vyloučí se z hledání
  a nahradí mediánem okolí. Změřeno: bez pojistky přehlasuje 60 vadných pixelů
  20 skutečných částic a posun vyjde (0,00; 0,00) místo (+8,0; −5,0).
- **Interpolace Lanczosem.** Bilineární interpolace snímek rozmaže a kolem každé
  statické částice vyrobí falešnou detekci (37 na poli 120 částic); Lanczos jen 3.
- **Ořez okraje** stejným podílem v obou osách (poměr stran zůstává). Ve výchozím
  režimu se ořezává jen podle naměřeného driftu, volitelně pevných 90 %.
- Kotvou je první snímek, ne předchozí — chyby se tak nesčítají. Srovnává se
  i referenční pozadí.

Měřeno na sérii s driftem 11 px, kde skutečně přibylo 60 částic:

| | částic na konci | pokrytí |
|---|---|---|
| bez driftu (správná odpověď) | 57 | 0,549 % |
| s driftem, bez zarovnání | **715** | **53,4 %** |
| s driftem, se zarovnáním | **57** | 0,618 % |

Zarovnání stojí asi 94 ms na snímek při binningu 2. Vypíná se zaškrtávátkem
v GUI nebo přepínačem `--no-align`.

### Barevné i černobílé snímky

Barevný snímek se převede na intenzitu jedním z režimů (`--mono`, v GUI
*Barevný snímek jako*):

| Režim | Kdy použít |
|-------|-----------|
| `luma` (výchozí) | vážený jas Rec.601 – stejný převod, jakým počítá mono kanál sama kamera, takže sedí na referenci `mean_mono` |
| `prumer` | nepodceňuje modrou; modravé rozptylové halo částic má plnou váhu |
| `maximum` | nejcitlivější na částice svítící jen v jednom kanálu, ale zvyšuje šum pozadí |
| `r` / `g` / `b` | cílené měření jednoho kanálu |

Mono snímky projdou beze změny. Snímky s alfa kanálem se převedou na BGR.

### Provoz

- **Paralelní zpracování** s ohledem na paměť: počet vláken se automaticky sníží,
  aby analýza notebook nevytlačila do odkládacího souboru.
- **Chyba u jednoho souboru sérii nezastaví** – skončí v seznamu chyb.
- **Pojistka proti vlastním výstupům.** Exportované grafy se ukládají do složky
  se snímky, takže při druhém spuštění hrozilo, že se použijí jako snímky měření –
  a protože se řadí abecedně před `df_00001_…`, staly by se z nich rovnou
  **bias snímky**. Filtrují se dvakrát: podle názvu (`analyza_*`, `*_grafy`,
  `*_nahled`, `*_souhrn`, …) a podle rozlišení (to se určí hlasováním z prvních
  osmi souborů; co mu neodpovídá, se vyřadí a vypíše). Druhá pojistka zachytí
  i cizí obrázek s nevinným názvem – vyřadí se ještě před sestavením časové osy,
  takže nezmění ani bias, ani časy snímků.
- **Okno se přizpůsobí displeji.** Velikost se odvozuje z volné plochy obrazovky
  (dřív byla pevných 1440×900, což je víc než Full HD se škálováním Windows 150 %,
  tedy 1280×720 logických bodů – spodek okna byl mimo obraz). Levý panel se
  roluje, uložená geometrie se ořízne na aktuální monitor.
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
2. **Referenční pozadí (bias)** – režim (automaticky / vždy ze složky / vždy ze
   série), volba složky s referencemi, zaškrtávátko *Srovnat úroveň reference se
   snímky* a modrý řádek s tím, **která reference se právě použije** a jak dlouho
   před měřením vznikla. Náhled se obnovuje hned po výběru složky. Tady je
   i *Srovnat drift sklíčka / kamery* a volba ořezu po srovnání.
3. **Parametry analýzy** – rozlišení/binning, počet a metoda bias snímků, převod
   barevného snímku na intenzitu, režim a hodnota prahu, práh zamlžení, hranice
   hotspotu, minimální plocha částice, hranice velkého shluku, protáhlost a délka
   vlákna, kalibrace µm/px, počet vláken CPU, ROI. Každé pole má nápovědu po
   najetí myší. Panel je v jednom sloupci a roluje se, takže se vejde i na nízký
   displej.
4. **Spuštění a export** – průběh, zastavení, export CSV + grafy + JSON.

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
- 📋 Datová tabulka · 📖 Průvodce s popisem všech veličin
- 🧾 **Souhrn měření** – první řádek uvádí, odkud pochází referenční pozadí
  (soubor reference a naměřený posun úrovně, nebo počet bias snímků ze série).
  Informativní poznámky k průběhu jsou oddělené od skutečných upozornění, takže
  běžný běh už nevyskakuje dialog.

---

## Příkazová řádka

```powershell
# jedna složka
py main.py --folder "C:\...\darkfield_20260907_145621" --bias 3 --binning 2 --scale 0.35

# všechny podsložky najednou
py main.py --folder "C:\...\BMS fotky" --batch

# jen výřez, absolutní práh, bez výpisu průběhu
py main.py --folder "C:\...\mereni" --roi 200,100,1500,900 --mode absolute --absolute 10 --quiet

# vynutit referenci ze složky a barevné snímky měřit průměrem kanálů
py main.py --folder "C:\...\mereni" --reference reference --mono prumer
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
| `--reference` | zdroj pozadí: `auto` / `reference` / `serie` | `auto` |
| `--reference-dir` | složka s `.npz` referencemi | hledá se `reference` |
| `--no-level-match` | nesrovnávat úroveň externí reference | srovnává se |
| `--mono` | převod barvy: `luma`/`prumer`/`maximum`/`r`/`g`/`b` | `luma` |
| `--no-align` | nezarovnávat snímky podle souhvězdí částic | zarovnává se |
| `--align-crop` | ořez po zarovnání: `auto`/`fixed`/`none` | `auto` |
| `--align-stars`, `--align-max-shift` | velikost souhvězdí a mez posunu | 100, 60 px |

---

## Metodika

> Tohle je jen shrnutí. Podrobný výklad s kódem, odvozením vzorců a obrázky
> najdete v [docs/JAK_TO_FUNGUJE.md](docs/JAK_TO_FUNGUJE.md).

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
alignment.py             srovnání driftu podle souhvězdí prachových částic
imageops.py              odhad šumu a separace oparu (sdílí analyzer i alignment)
frameio.py               načítání snímků, Unicode cesty, časová razítka, převod barvy
reference.py             hledání, výběr a načtení reference z .npz + .json
exporter.py              CSV, souhrnné grafy, JSON souhrn
gui.py                   grafické rozhraní (PyQt6)
viewer.py                prohlížeč snímků s klasifikační maskou
help_text.py             text nápovědy zobrazený v GUI
main.py                  spuštění GUI i dávkové analýzy z příkazové řádky
tools/make_demo_series.py generátor ukázkové série
tools/make_docs_figures.py generátor obrázků do dokumentace
tests/                   automatické testy (pytest)
docs/JAK_TO_FUNGUJE.md   podrobný výklad analýzy (jak a proč to funguje)
docs/img/                obrázky do dokumentace
```

## Testy

```powershell
py -m pip install pytest
py -m pytest -q
```

120 testů pokrývá jádro, zarovnání driftu, výběr reference, barevné snímky,
export i grafické rozhraní (běží bez obrazovky přes `QT_QPA_PLATFORM=offscreen`), včetně regresí
na všechny tři výše popsané pády.
