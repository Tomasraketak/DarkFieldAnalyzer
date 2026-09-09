"""Text nápovědy zobrazený v záložce „Průvodce“ (oddělen od kódu GUI)."""

HELP_HTML = """
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #222; }
    h2 { color: #1B4F72; border-bottom: 2px solid #2980B9; padding-bottom: 4px; margin-top: 20px; }
    h3 { color: #2C3E50; margin-top: 15px; margin-bottom: 5px; }
    .box { background-color: #F8F9FA; border-left: 4px solid #2980B9; padding: 10px 15px; margin: 10px 0; }
    .tip { background-color: #E8F8F5; border-left: 4px solid #27AE60; padding: 10px 15px; margin: 10px 0; }
    .warn { background-color: #FEF9E7; border-left: 4px solid #F39C12; padding: 10px 15px; margin: 10px 0; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
    th, td { border: 1px solid #BDC3C7; padding: 6px 10px; text-align: left; }
    th { background-color: #EAEDED; }
    code { background: #EEE; padding: 1px 4px; }
    b.blue { color: #0088CC; } b.yellow { color: #B7950B; } b.red { color: #C0392B; }
    b.green { color: #27AE60; } b.purple { color: #8E44AD; }
</style>

<h2>📖 Průvodce a vědecký popis analýz</h2>
<p>Aplikace vyhodnocuje časové řady z mikroskopie v temném poli (<i>dark-field</i>).
V temném poli je čistý povrch sklíčka černý; každá nečistota, prach nebo kondenzát
rozptyluje světlo do objektivu a září.</p>

<div class="tip">
<b>Rychlý postup:</b> vlevo vyberte složku se snímky → zkontrolujte parametry →
<b>SPUSTIT ANALÝZU</b> → výsledky v záložkách vpravo → <b>Exportovat výsledky</b>.
</div>

<h3>1. Referenční bias (odečet pozadí)</h3>
<div class="box">
    <b>Co dělá:</b> Od každého snímku odečte referenční mapu pozadí:
    <code>diference = snímek − bias</code>.<br>
    <b>Co filtruje:</b> prach usazený na optice mikroskopu, horké pixely senzoru,
    nerovnoměrné osvětlení. Měříte tedy <b>jen nově vzniklou kontaminaci</b>.<br>
    <b>Odkud pozadí pochází</b> (panel „1b. Referenční pozadí“):<br>
    • <b>Automaticky</b> (výchozí) – použije se reference uložená programem
    BMS Cam Control ve složce <code>reference</code>; když žádná není, spočítá se
    bias z prvních snímků série.<br>
    • <b>Vždy ze složky s referencemi</b> – bez reference analýza skončí chybou.
    Hodí se, když chcete mít jistotu, že se neměří proti prvním snímkům měření.<br>
    • <b>Vždy z prvních snímků série</b> – původní chování, reference se ignoruje.<br>
    <b>Metoda</b> (jen pro bias ze série): <i>medián</i> (výchozí) je odolný – jedna
    náhodná částice v bias snímku referenci nezkazí. <i>Průměr</i> je o něco tišší,
    ale citlivý na výjimečné hodnoty. 3–5 snímků je dobrý kompromis.<br>
    <b>Pozor:</b> snímky, ze kterých se bias počítá, se porovnávají samy se sebou,
    takže se z výsledné řady vynechávají. <b>Externí reference tuto daň neplatí</b> –
    zpracují se všechny snímky ve složce.
</div>

<h3>1b. Reference ze složky <code>reference</code></h3>
<div class="box">
    <b>Formát:</b> dvojice souborů, které ukládá záznamový program:
    <code>reference_20260908_142910.npz</code> (průměr N tmavých snímků v poli
    <code>mean_mono</code>) a stejnojmenný <code>.json</code> s nastavením kamery
    a osvětlení. JSON slouží jen k popisu, analýza z něj nic nepřebírá.<br>
    <b>Kde se hledá:</b> podsložka <code>reference</code> nejdřív přímo u měření,
    pak v nadřazených složkách – typicky
    <code>…\\BMS fotky\\reference</code> vedle složek s měřeními. Tlačítkem
    <b>Složka…</b> se dá určit ručně, tlačítkem <b>Auto</b> zrušit zpět.<br>
    <b>Která se vybere:</b> ta, která vznikla <b>naposledy před začátkem měření</b>.
    Čas se čte z názvu souboru, takže se kvůli výběru nemusí rozbalovat obrazová data.
    Pozdější reference nikdy nevyhraje, ani když je časově blíž – popisovala by
    pozadí, které v době měření ještě neplatilo. Když existují jen pozdější,
    použije se nejbližší z nich a analýza to ohlásí jako upozornění.<br>
    <b>Srovnat úroveň reference se snímky:</b> reference vznikla v jiném okamžiku,
    takže se od snímků může lišit konstantním posunem jasu (teplota senzoru, jas
    zdroje). Bez srovnání by se takový posun projevil jako <b>plošné zamlžení přes
    celý snímek</b>. Posun se odhaduje <b>jednou pro celou sérii</b> jako nejnižší
    naměřená úroveň pozadí – v temném poli totiž kontaminace jas jen přidává, takže
    nejtmavší snímek je nejlepší odhad skutečné nuly. Kdyby se počítal pro každý
    snímek zvlášť, odečetlo by se i skutečné zamlžení. Daní je, že se odečte
    i kontaminace, která je po celou sérii úplně stejná; posun menší než 0,5 ADU
    se ignoruje, takže dobře sedící reference se nijak „neopravuje“. Skutečnou
    hodnotu najdete v souhrnu i v exportovaném JSON.
</div>

<h3>1d. Zarovnání driftu sklíčka</h3>
<div class="box">
    <b>Proč:</b> Během dlouhého měření se sklíčko nebo kamera posune o jednotky až
    desítky pixelů. Statické částice se pak přestanou krýt s referencí a zůstane
    po nich <b>světlý půlměsíc</b>, který analýza počítá jako novou kontaminaci.
    Změřeno na sérii s driftem 11 px: bez zarovnání 715 částic, se zarovnáním 57 –
    a 57 je správná odpověď (stejná série bez driftu).<br>
    <b>Jak:</b> Ze snímku se vybere 100 nejjasnějších prachových částic jako
    „souhvězdí“ a pro každou dvojici částic se hlasuje o posunu. Skutečný posun
    dostane tolik hlasů, kolik je společných částic; náhodné dvojice se rozptýlí.
    Přesnost je lepší než desetina pixelu. Kotvou je vždy <b>první snímek</b>, ne
    předchozí – chyby se tak nesčítají.<br>
    <b>Vadné pixely:</b> Horké pixely senzoru se s driftem nepohybují, takže by
    hlasovaly pro nulový posun. Poznají se podle toho, že jejich sousedé jsou na
    úrovni pozadí (skutečná částice je rozmazaná optikou), vyloučí se z hledání
    a nahradí mediánem okolí.<br>
    <b>Ořez:</b> Po srovnání chybí u každého snímku pruh na okraji. Ořezává se
    stejným podílem v obou osách, takže <b>poměr stran zůstává</b>.
    <i>Podle driftu</i> (výchozí) ořízne jen tolik, kolik je nutné – u klidného
    měření desetiny procenta. <i>Pevných 90 %</i> dá stejnou plochu bez ohledu na
    drift, což se hodí, když chcete porovnávat různá měření mezi sebou.<br>
    <b>Cena:</b> asi 94 ms na snímek při binningu 2. Kdyby se nenašlo dost
    výrazných částic, zarovnání se tiše vypne a napíše to v poznámkách.
</div>

<h3>1c. Barevné a černobílé snímky</h3>
<div class="box">
    Aplikace zpracuje obojí. Barevný snímek se převede na intenzitu podle volby
    <b>Barevný snímek jako</b>:<br>
    • <b>Vážený jas – Rec.601</b> (výchozí, doporučeno) – standardní černobílý převod,
    stejný, jakým počítá mono kanál sama kamera. Sedí proto na referenci
    <code>mean_mono</code>.<br>
    • <b>Průměr kanálů</b> – nepodceňuje modrou, takže modravé rozptylové halo částic
    v temném poli má plnou váhu.<br>
    • <b>Maximum kanálů</b> – nejcitlivější na částice svítící jen v jednom kanálu,
    ale zvyšuje i šum pozadí.<br>
    • <b>Jen červený / zelený / modrý kanál</b> – pro cílené měření jedné barvy.<br>
    Pokud změníte volbu a reference zůstane mono, může vzniknout posun úrovně –
    postará se o něj srovnání popsané výše, ale hodnoty pak nejsou přímo srovnatelné
    s měřením v jiném režimu.
</div>

<h3>2. Difuzní zamlžení a kondenzace (Haze) – <b class="blue">azurová</b></h3>
<div class="box">
    <b>Co dělá:</b> Odděluje nízkofrekvenční složku jasu, tedy hladký rozptyl světla.
    Odhad se počítá na silně zmenšeném obraze s morfologickým otevřením, takže jasné
    částice odhad oparu neznečistí.<br>
    <b>Co vidí:</b> zapaření dechem, orosení, kondenzaci vlhkosti, tenký kapalný film.<br>
    <b>Parametr:</b> <b>Práh zamlžení [ADU]</b> (výchozí 4,0). Pro sotva viditelný závoj
    snižte na 2–3, při kolísajícím osvětlení zvyšte na 5–7.
</div>

<h3>3. Mikročástice – <b class="yellow">žlutá</b></h3>
<div class="box">
    <b>Co dělá:</b> Prahuje ostrou složku (<code>diference − opar</code>) a označuje
    kompaktní objekty menší než hranice velkého shluku.<br>
    <b>Parametr:</b> <b>Min. plocha [px]</b> (výchozí 3 px) filtruje ojedinělé šumové pixely.
    Hodnota se zadává v pixelech <i>plného rozlišení</i>, takže při binningu nemusíte nic přepočítávat.
</div>

<h3>4. Velké shluky a kapky – <b class="red">červená</b></h3>
<div class="box">
    Souvislé objekty s plochou nad <b>Shluk od [px]</b> (výchozí 100 px): agregovaný prach,
    kapky tekutiny, usazeniny.
</div>

<h3>5. Vlákna a škrábance – <b class="green">zelená</b></h3>
<div class="box">
    <b>Co dělá:</b> Protáhlost se počítá z <b>momentů druhého řádu</b> (poměr os
    ekvivalentní elipsy), nikoliv z opsaného obdélníku. Šikmé vlákno pod 45° má opsaný
    obdélník téměř čtvercový – dřívější způsob ho proto klasifikoval jako shluk.<br>
    <b>Parametry:</b> <b>Protáhlost</b> (výchozí 2,8) a <b>Min. délka</b> (výchozí 12 px).
</div>

<h3>6. Hotspoty – <b class="purple">fialová</b></h3>
<div class="box">
    Pixely s jasem nad hranicí <b>Hotspot od [ADU]</b> (výchozí 250), tedy prakticky
    saturovaný senzor: zrcadlové odlesky, kovové nebo dielektrické částice.
    Vysoký počet hotspotů znamená, že měření intenzity je v těchto místech nelineární.
</div>

<h3>7. Nehomogenita a těžiště kontaminace (✚)</h3>
<div class="box">
    Zorné pole se rozdělí na mřížku 8×8 zón a spočítá se variační koeficient pokrytí,
    normovaný svým maximem – hodnota je tedy skutečně v rozsahu 0–100 %.<br>
    • <b>0–25 %:</b> kontaminace je rozprostřená rovnoměrně (plošný opar, jemný prach).<br>
    • <b>nad 50 %:</b> lokální znečištění (kapka, otisk prstu u kraje, jeden velký škrábanec).<br>
    • <b>Těžiště [X %, Y %]</b> ukazuje střed hmoty nečistot, v prohlížeči vyznačený křížem.
</div>

<h3>8. Skóre čistoty (0–100 %)</h3>
<div class="tip">
    Souhrnné číslo složené ze tří saturujících penalizací: pokrytí plochy (45 bodů),
    průměrný jas oparu (30 bodů) a hustota částic na megapixel (25 bodů).
    Hustota se počítá na megapixel, takže skóre nezávisí na rozlišení kamery ani na binningu.<br>
    • <b>95–100:</b> excelentní stav &nbsp;•&nbsp; <b>80–95:</b> mírná kontaminace &nbsp;•&nbsp;
    <b>pod 80:</b> znečištěné sklíčko.
</div>

<h3>9. Složení kontaminace (záložka 🧩)</h3>
<div class="box">
    <b>Co ukazuje:</b> kolik procent plochy snímku zabírá který typ kontaminace, v čase.
    Vrstvy jsou <b>disjunktní</b> – zamlžení se počítá bez plochy, na které leží částice –
    takže jejich součet je přesně celkové pokrytí.<br>
    <b>Proč jsou grafy dva:</b> zamlžení se měří jako plocha nad prahem 4 ADU, takže při
    zapaření zabere klidně 90 % snímku, zatímco částice bývají v desetinách procenta.
    Druhý graf proto ukazuje totéž <i>bez zamlžení</i> ve vlastním měřítku.<br>
    <b>Pruh dole:</b> průměrné zastoupení typů v kontaminované ploše (dohromady 100 %).<br>
    Barvy odpovídají barvám v prohlížeči snímků. V CSV najdete sloupce
    <code>Pokrytí zamlžením bez částic [%]</code>, <code>Pokrytí mikročásticemi [%]</code>,
    <code>Pokrytí shluky [%]</code> a <code>Pokrytí vlákny [%]</code>.
</div>

<h3>10. Ostrost a šum pozadí (kontrola kvality měření)</h3>
<div class="box">
    <b>Ostrost</b> (variance Laplaciánu) prudce klesne, pokud se mikroskop rozostří nebo
    dojde k otřesu – takový snímek nemá smysl porovnávat se zbytkem řady.<br>
    <b>Šum pozadí σ</b> ukazuje, jak stabilní byla expozice; z něj se odvozuje práh
    detekce (<code>práh = medián + sigma × σ</code>).
</div>

<h3>11. Rychlost změny (d/dt) a fáze děje</h3>
<div class="warn">
    Derivace se počítá lokální lineární regresí přes 5 snímků, ne rozdílem sousedních –
    u rychlých sérií tak nezesiluje šum.<br>
    • <b class="red">Nárůst / zamlžování:</b> kontaminace přibývá (moment zafoukání, kondenzace).<br>
    • <b class="blue">Odpařování / ústup:</b> opar mizí, povrch osychá.<br>
    • <b class="green">Stabilní:</b> beze změny nad úrovní šumu.
</div>

<h3>12. Pojistky proti cizím souborům ve složce</h3>
<div class="box">
    Export ukládá grafy do stejné složky, kde jsou snímky. Aby se z nich při
    dalším spuštění nestaly „snímky měření“ (a hlavně ne referenční bias, protože
    se řadí abecedně před <code>df_00001_…</code>), aplikace hlídá dvě věci:<br>
    • <b>Název:</b> soubory začínající <code>analyza_</code> a soubory s příponou
    <code>_grafy</code>, <code>_nahled</code>, <code>_souhrn</code>, <code>_overlay</code>
    nebo <code>_maska</code> se ignorují.<br>
    • <b>Rozlišení:</b> rozlišení měření se určí hlasováním z prvních osmi souborů;
    co mu neodpovídá, se vyřadí a vypíše v upozornění po skončení analýzy.
    Tím se odfiltruje i cizí obrázek s nevinným názvem.<br>
    Vyřazené soubory najdete v okně „Upozornění k analýze“ i ve výpisu na příkazové řádce.
</div>

<h3>13. Co dělat, když výsledky nesedí</h3>
<table>
<tr><th>Projev</th><th>Řešení</th></tr>
<tr><td>Tisíce „částic“ i na čistém sklíčku</td>
    <td>Zvyšte <b>Sigma</b> na 5–6 nebo <b>Min. plochu</b> na 5 px. Zkontrolujte, zda první
    snímky použité pro bias byly opravdu čisté.</td></tr>
<tr><td>Nezachytí se jemný opar</td><td>Snižte <b>Práh zamlžení</b> na 2–3 ADU.</td></tr>
<tr><td>Shluky se hlásí jako vlákna</td><td>Zvyšte <b>Protáhlost</b> na 3,5.</td></tr>
<tr><td>Analýza je pomalá</td>
    <td>Zvolte binning 2×2, případně zvyšte počet vláken CPU (0 = automaticky).</td></tr>
<tr><td>Plochy v µm² jsou nesmyslné</td><td>Nastavte správné <b>Měřítko [µm/px]</b> podle kalibrace objektivu.</td></tr>
<tr><td>Celá plocha se hlásí jako zamlžená</td>
    <td>Reference nesedí na snímky. Zapněte <b>Srovnat úroveň reference se snímky</b>
    a zkontrolujte v souhrnu, kterou referenci analýza vzala a jaký posun naměřila.</td></tr>
<tr><td>Použila se jiná reference, než jsem čekal</td>
    <td>Vybírá se poslední pořízená <i>před</i> měřením. Zkontrolujte čas v názvu
    souboru <code>reference_RRRRMMDD_HHMMSS.npz</code>; složku lze určit i ručně.</td></tr>
<tr><td>Najednou skokově vyskočí počet částic</td>
    <td>Zkontrolujte v souhrnu <b>Drift scény</b>. Když je velký a zarovnání
    hlásí málo spárovaných částic, vzorek se pravděpodobně posunul víc, než
    stačí sledovat – zkontrolujte upevnění.</td></tr>
<tr><td>Analýza je pomalejší než dřív</td>
    <td>Zarovnání stojí asi 94 ms na snímek. Když se vzorek prokazatelně nehýbe,
    dá se vypnout zaškrtávátkem <b>Srovnat drift sklíčka / kamery</b>.</td></tr>
<tr><td>Barevné snímky vycházejí jinak než mono</td>
    <td>Zkontrolujte volbu <b>Barevný snímek jako</b> – na referenci
    <code>mean_mono</code> sedí <i>Vážený jas (Rec.601)</i>.</td></tr>
</table>
"""
