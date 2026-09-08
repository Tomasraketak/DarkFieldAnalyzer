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
    <b>Co dělá:</b> Z prvních <i>N</i> snímků sestaví referenční mapu pozadí a tu odečte
    od každého dalšího snímku: <code>diference = snímek − bias</code>.<br>
    <b>Co filtruje:</b> prach usazený na optice mikroskopu, horké pixely senzoru,
    nerovnoměrné osvětlení. Měříte tedy <b>jen nově vzniklou kontaminaci</b>.<br>
    <b>Metoda:</b> <i>medián</i> (výchozí) je odolný – jedna náhodná částice v bias snímku
    referenci nezkazí. <i>Průměr</i> je o něco tišší, ale citlivý na výjimečné hodnoty.<br>
    <b>Kolik snímků:</b> 3–5 je dobrý kompromis. Při 1 snímku se do reference propíše
    i jeho vlastní šum, takže prahy vycházejí méně stabilní.
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

<h3>9. Ostrost a šum pozadí (kontrola kvality měření)</h3>
<div class="box">
    <b>Ostrost</b> (variance Laplaciánu) prudce klesne, pokud se mikroskop rozostří nebo
    dojde k otřesu – takový snímek nemá smysl porovnávat se zbytkem řady.<br>
    <b>Šum pozadí σ</b> ukazuje, jak stabilní byla expozice; z něj se odvozuje práh
    detekce (<code>práh = medián + sigma × σ</code>).
</div>

<h3>10. Rychlost změny (d/dt) a fáze děje</h3>
<div class="warn">
    Derivace se počítá lokální lineární regresí přes 5 snímků, ne rozdílem sousedních –
    u rychlých sérií tak nezesiluje šum.<br>
    • <b class="red">Nárůst / zamlžování:</b> kontaminace přibývá (moment zafoukání, kondenzace).<br>
    • <b class="blue">Odpařování / ústup:</b> opar mizí, povrch osychá.<br>
    • <b class="green">Stabilní:</b> beze změny nad úrovní šumu.
</div>

<h3>11. Co dělat, když výsledky nesedí</h3>
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
</table>
"""
