"""Per-language in-context-learning exemplar blocks.

Extracted verbatim from the original build_gec_representation_cache.py
and retrieve_gec_examples_by_representation.py. The two scripts had
identical copies; this module is the single source.

Each ICL block is a string of alternating <erroneous sentence>...
</erroneous sentence>\\n<corrected sentence>...</corrected sentence>\\n
pairs, ready to be slotted into the prompt template.

For Chinese (zh), the original code uses no ICL block (empty string).
"""
from __future__ import annotations

ICL_EN = (
    "<erroneous sentence> Suddenly I saw a small gap between two blocks of snowy hills just enough for me to lie down in .</erroneous sentence>\n"
    "<corrected sentence> Suddenly I saw a small gap between two blocks of snowy hills just big enough for me to lie down in .</corrected sentence>\n"
    "<erroneous sentence> In addition , we are sorry that your holiday experience was affected by the fire that occurred on Saturday October 10th , 2015 .</erroneous sentence>\n"
    "<corrected sentence> In addition , we are sorry that your holiday experience was affected by the fire that occurred on Saturday October 10th , 2015 .</corrected sentence>\n"
    "<erroneous sentence> Even if the room counts with a considerable number of computers , the absence of Internet does not give students the opportunity to use the many websites useful to improve their language skills .</erroneous sentence>\n"
    "<corrected sentence> Even if the room has a considerable number of computers , the absence of internet access does not give students the opportunity to use the many websites that would be useful to improve their language skills .</corrected sentence>\n"
    "<erroneous sentence> We managed to raise much more money for a good cause than we hoped to raise .</erroneous sentence>\n"
    "<corrected sentence> We managed to raise much more money for a good cause than we hoped to raise .</corrected sentence>\n"
    "<erroneous sentence> Many people do n't care about wasting their time even when they busy .</erroneous sentence>\n"
    "<corrected sentence> Many people do n't care about wasting their time even when they are busy .</corrected sentence>\n"
    "<erroneous sentence> That is why it is not to be the key point that considered the key to success .</erroneous sentence>\n"
    "<corrected sentence> That is why it should not be the key point that is considered the key to success .</corrected sentence>\n"
    "<erroneous sentence> For instance , some group could have questions , but it would not be able to ask the other group because they would sleep at that time .</erroneous sentence>\n"
    "<corrected sentence> For instance , some group could have questions , but it would not be able to ask the other group because they would be sleeping at that time .</corrected sentence>\n"
    "<erroneous sentence> The people who eager to play football must follow some basic rules which are described below :</erroneous sentence>\n"
    "<corrected sentence> People who are eager to play football must follow some basic rules which are described below :</corrected sentence>\n"
)

ICL_DE = (
    "<erroneous sentence> Ich möchte zwei Zimmer hell , groß Ein Balkon auf Ich habe ein paar fragen : Wie hoch sind die Nebenkosten und die Kaution .</erroneous sentence>\n"
    "<corrected sentence> Ich möchte zwei helle , große Zimmer , einen Balkon . Ich habe ein paar Fragen : Wie hoch sind die Nebenkosten und die Kaution ?</corrected sentence>\n"
    "<erroneous sentence> Ebenso uralt sind ihre Träume einer erfolgreichen Karriere , in der sie ebenso viel Geld als Männer verdienen könnte .</erroneous sentence>\n"
    "<corrected sentence> Ebenso uralt sind ihre Träume einer erfolgreichen Karriere , in der sie ebenso viel Geld wie Männer verdienen könnten .</corrected sentence>\n"
    "<erroneous sentence> Es ist sinnlos eine ganze Sprache zu ändern , aber es ist wichtig zu merken wie männlich orientiert so eine ganztage Ding wie Kommunikation scheint .</erroneous sentence>\n"
    "<corrected sentence> Es ist sinnlos eine ganze Sprache zu ändern , aber es ist wichtig , zu merken , wie männlich-orientiert so ein ganztags Ding wie Kommunikation scheint .</corrected sentence>\n"
    "<erroneous sentence> Die Firmen wollen nicht Teorispezialisten mit hohen Noten haben !</erroneous sentence>\n"
    "<corrected sentence> Die Firmen wollen keine Theoriespezialisten mit hohen Noten haben !</corrected sentence>\n"
    "<erroneous sentence> Daher leistet man seinen Beitrag nicht für die Gesellschaft , sondern für sich selbst bzw. für seine Familie ;</erroneous sentence>\n"
    "<corrected sentence> Daher leistet man seinen Beitrag nicht für die Gesellschaft , sondern für sich selbst bzw. für seine Familie ;</corrected sentence>\n"
    "<erroneous sentence> Ich muss viel arbeiten , aber jetzt bin ich glücklich !</erroneous sentence>\n"
    "<corrected sentence> Ich muss viel arbeiten , aber jetzt bin ich glücklich !</corrected sentence>\n"
    "<erroneous sentence> Glücklicherweise funktionieren die Verkehrsmittel sehr gut , und sie sind billig .</erroneous sentence>\n"
    "<corrected sentence> Glücklicherweise funktionieren die Verkehrsmittel sehr gut , und sie sind billig .</corrected sentence>\n"
    "<erroneous sentence> Ohne genug Ärzte könnte eine große Bevölkerung nicht existieren , denn zu viele Menschen würden wegen Krankheit als Babys oder als Kinder sterben .</erroneous sentence>\n"
    "<corrected sentence> Ohne genug Ärzte könnte eine große Bevölkerung nicht existieren , denn zu viele Menschen würden wegen Krankheit als Babys oder als Kinder sterben .</corrected sentence>\n"
)

ICL_RU = (
    "<erroneous sentence> Если БАДы нужны человеку как эликсир молодости , то тогда они могут понадобиться в юном возрасте .</erroneous sentence>\n"
    "<corrected sentence> Если БАДы нужны человеку , как эликсир молодости , то тогда они могут понадобиться в юном возрасте .</corrected sentence>\n"
    "<erroneous sentence> Конец рассказа говорит не только о нем , но и о религии .</erroneous sentence>\n"
    "<corrected sentence> Конец рассказа говорит не только о нем , но и о религии .</corrected sentence>\n"
    "<erroneous sentence> В дополнение к матрёшке , когда я был в России , мой отец попросил меня купить ему ушанку с советским символом , потому что\n он русский на половину .</erroneous sentence>\n"
    "<corrected sentence> В дополнение к матрёшке , когда я был в России , мой отец попросил меня купить ему ушанку с советским символом , потому что\n он русский на половину .</corrected sentence>\n"
    "<erroneous sentence> В советское время экономика состояла из промышленности , сельского хозяйства , военной технологии и других секторов .</erro\nneous sentence>\n"
    "<corrected sentence> В советское время экономика состояла из промышленности , сельского хозяйства , военной технологии и других секторов .</corr\nected sentence>\n"
    "<erroneous sentence> В большинстве эмигрантами были военные , дворяне , интеллигенция , профессионалы , казаки и духовенство , государственные с\nлужащие , а также члены их семей .</erroneous sentence>\n"
    "<corrected sentence> В большинстве эмигрантами были военные , дворяне , интеллигенция , профессионалы , казаки и духовенство , государственные с\nлужащие , а также члены их семей .</corrected sentence>\n"
    "<erroneous sentence> В тот вечер Шарль сказал его первое слово за три года .</erroneous sentence>\n"
    "<corrected sentence> В тот вечер Шарль сказал свое первое слово за три года .</corrected sentence>\n"
    "<erroneous sentence> Употребление таких лёгких наркотиков , как марихуана , может считаться преступлением без жертв , но всё равно даже само нал\nичие маленького количества марихуаны является уголовным преступлением в большинстве штатов США .</erroneous sentence>\n"
    "<corrected sentence> Употребление таких лёгких наркотиков , как марихуана , может считаться преступлением без жертв , но всё равно даже само нал\nичие маленького количества марихуаны является уголовным преступлением в большинстве штатов США .</corrected sentence>\n"
    "<erroneous sentence> Вы можете на образовательный обмен ехать за границу и учиться .</erroneous sentence>\n"
    "<corrected sentence> Вы можете ехать за границу и учиться по образовательному обмену .</corrected sentence>\n"
)

ICL_ET = (
    "<erroneous sentence> Isegi tudengid peavad nüüd enamus oma töödest esitama trükituna .</erroneous sentence>\n"
    "<corrected sentence> Isegi tudengid peavad nüüd enamuse oma töödest esitama trükituna .</corrected sentence>\n"
    "<erroneous sentence> Selles mõisas saab veel näha üle 300 erinevaid põõsaid .</erroneous sentence>\n"
    "<corrected sentence> Selles mõisas saab veel näha üle 300 erineva põõsa .</corrected sentence>\n"
    "<erroneous sentence> Selge on seda , et venelased ja eestlased vähe suhtlevad omavahel .</erroneous sentence>\n"
    "<corrected sentence> Selge on see , et venelased ja eestlased suhtlevad omavahel vähe .</corrected sentence>\n"
    "<erroneous sentence> Selleks , et soovitatud eesmärgi saavutada on vaja paljudest raskustest ja takistustest üle saada .</erroneous sentence>\n"
    "<corrected sentence> Selleks , et soovitud eesmärki saavutada , on vaja paljudest raskustest ja takistustest üle saada .</corrected sentence>\n"
    "<erroneous sentence> Ma tulin täna hommikul , nagu te teate , üsna vara siia ja teel siia nägin ma vähemalt kümmet meest , kes olid purjus ja räpased .</erroneous sentence>\n"
    "<corrected sentence> Ma tulin täna hommikul , nagu te teate , üsna vara siia ja teel siia nägin ma vähemalt kümmet meest , kes olid purjus ja räpased .</corrected sentence>\n"
    "<erroneous sentence> Aga elus on peaaegu kõik tasuline , me peame kõige eest maksma .</erroneous sentence>\n"
    "<corrected sentence> Aga elus on peaaegu kõik tasuline , me peame kõige eest maksma .</corrected sentence>\n"
    "<erroneous sentence> Tema väga hästi aru saanud sellest . et mida tähendab termin ' sociare ' ( lad keeles - kokku koguma ) .</erroneous sentence>\n"
    "<corrected sentence> Tema väga hästi aru saanud sellest . et mida tähendab termin ' sociare ' ( lad keeles - kokku koguma ) .</corrected sentence>\n"
    "<erroneous sentence> Mina arvan , et see on väga hea raamat selleks , et aru saada kuidas hääldata häälikuid ja sõnu .</erroneous sentence>\n"
    "<corrected sentence> Mina arvan , et see on väga hea raamat selleks , et aru saada kuidas hääldata häälikuid ja sõnu .</corrected sentence>\n"
)

ICL_RO = (
    "<erroneous sentence> Industria dronelor stă să explodeze în Uniunea Europeană în următori ani .</erroneous sentence>\n"
    "<corrected sentence> Industria dronelor stă să explodeze în Uniunea Europeană în următorii ani .</corrected sentence>\n"
    "<erroneous sentence> va depune o contestație la Curtea Constituțională exact pe acest proiect de lege</erroneous sentence>\n"
    "<corrected sentence> va depune o contestație la Curtea Constituțională exact în legătură cu acest proiect de lege</corrected sentence>\n"
    "<erroneous sentence> A spus că la sfârșitul lui august când sa întors în SUA imediat după incidentul din tren , își dorea doar să se relaxeze alături de familie .</erroneous sentence>\n"
    "<corrected sentence> A spus că la sfârșitul lui august , când s-a întors în SUA imediat după incidentul din tren , își dorea doar să se relaxeze alături de familie .</corrected sentence>\n"
    "<erroneous sentence> Ce se-ntâmplă cu oamenii aceia care rulează pe ecran?</erroneous sentence>\n"
    "<corrected sentence> Ce se-ntâmplă cu oamenii aceia ale căror poze rulează pe ecran?</corrected sentence>\n"
    "<erroneous sentence> Van Kleeck spune că Aerojet plănuia începerea testării motoarelor la scară largă în 2017 , urmată apoi de obținerea certificatului în 2019 , însă data ar putea fi amânată în cazul în care compania nu primește suficiente fonduri de la Forțele Aeriene în contracte așteptate la sfârșitul primului semestru al anului fiscal 2016 , care începe în octombrie .</erroneous sentence>\n"
    "<corrected sentence> Van Kleeck spune că Aerojet plănuia începerea testării motoarelor la scară largă în 2017 , urmată apoi de obținerea certificatului în 2019 , însă data ar putea fi amânată în cazul în care compania nu primește suficiente fonduri de la Forțele Aeriene în contracte așteptate la sfârșitul primului semestru al anului fiscal 2016 , care începe în octombrie .</corrected sentence>\n"
    "<erroneous sentence> Vă rog să vă puneți la punct întâi cu descrierea/licențierea și apoi cu limitele folosirii imaginilor sub clauza utilizării cinstite ( de obicei se admite una singură din acest tip într-un articol ) .</erroneous sentence>\n"
    "<corrected sentence> Vă rog să vă puneți la punct întâi cu descrierea/licențierea și apoi cu limitele folosirii imaginilor sub clauza utilizării cinstite ( de obicei se admite una singură din acest tip într-un articol ) .</corrected sentence>\n"
    "<erroneous sentence> În anul următor a invadat Portugalia , căpătând regiunea Algarve .</erroneous sentence>\n"
    "<corrected sentence> În anul următor a invadat Portugalia , căpătând regiunea Algarve .</corrected sentence>\n"
    "<erroneous sentence> A declarat că Bulgaria este o țară cu o istorie bogată , pe care studiile recente au plasat-o pe locul al treilea , după Italia și Grecia , în topul traficului de obiecte și bunuri culturale .</erroneous sentence>\n"
    "<corrected sentence> A declarat că Bulgaria este o țară cu o istorie bogată , pe care studiile recente au plasat-o pe locul al treilea , după Italia și Grecia , în topul traficului de obiecte și bunuri culturale .</corrected sentence>\n"
)


_ICL_BY_LANG = {
    "en": ICL_EN,
    "bea19": ICL_EN,
    "de": ICL_DE,
    "ru": ICL_RU,
    "et": ICL_ET,
    "ro": ICL_RO,
    "zh": "",
}


def get_icl(lang: str) -> str:
    """Return the ICL exemplar block for ``lang``.

    Raises KeyError on an unrecognised language so callers fail loudly
    instead of silently using an English default.
    """
    if lang not in _ICL_BY_LANG:
        raise KeyError(
            f"Unsupported LANG={lang!r}; expected one of {sorted(_ICL_BY_LANG)}"
        )
    return _ICL_BY_LANG[lang]
