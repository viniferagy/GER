import os


ICL_TEMPLATES = {
    'reproduce_space_en_8fix': {
        'system': "",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\n{icl_examples}<erroneous sentence> Suddenly I saw a small gap between two blocks of snowy hills just enough for me to lie down in .</erroneous sentence>\n<corrected sentence> Suddenly I saw a small gap between two blocks of snowy hills just big enough for me to lie down in .</corrected sentence>\n<erroneous sentence> In addition , we are sorry that your holiday experience was affected by the fire that occurred on Saturday October 10th , 2015 .</erroneous sentence>\n<corrected sentence> In addition , we are sorry that your holiday experience was affected by the fire that occurred on Saturday October 10th , 2015 .</corrected sentence>\n<erroneous sentence> Even if the room counts with a considerable number of computers , the absence of Internet does not give students the opportunity to use the many websites useful to improve their language skills .</erroneous sentence>\n<corrected sentence> Even if the room has a considerable number of computers , the absence of internet access does not give students the opportunity to use the many websites that would be useful to improve their language skills .</corrected sentence>\n<erroneous sentence> We managed to raise much more money for a good cause than we hoped to raise .</erroneous sentence>\n<corrected sentence> We managed to raise much more money for a good cause than we hoped to raise .</corrected sentence>\n<erroneous sentence> Many people do n't care about wasting their time even when they busy .</erroneous sentence>\n<corrected sentence> Many people do n't care about wasting their time even when they are busy .</corrected sentence>\n<erroneous sentence> That is why it is not to be the key point that considered the key to success .</erroneous sentence>\n<corrected sentence> That is why it should not be the key point that is considered the key to success .</corrected sentence>\n<erroneous sentence> For instance , some group could have questions , but it would not be able to ask the other group because they would sleep at that time .</erroneous sentence>\n<corrected sentence> For instance , some group could have questions , but it would not be able to ask the other group because they would be sleeping at that time .</corrected sentence>\n<erroneous sentence> The people who eager to play football must follow some basic rules which are described below :</erroneous sentence>\n<corrected sentence> People who are eager to play football must follow some basic rules which are described below :</corrected sentence>\n<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence> {target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'reproduce_space_de_8fix': {
        'system': "",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\n{icl_examples}<erroneous sentence> Ich möchte zwei Zimmer hell , groß Ein Balkon auf Ich habe ein paar fragen : Wie hoch sind die Nebenkosten und die Kaution .</erroneous sentence>\n<corrected sentence> Ich möchte zwei helle , große Zimmer , einen Balkon . Ich habe ein paar Fragen : Wie hoch sind die Nebenkosten und die Kaution ?</corrected sentence>\n<erroneous sentence> Ebenso uralt sind ihre Träume einer erfolgreichen Karriere , in der sie ebenso viel Geld als Männer verdienen könnte .</erroneous sentence>\n<corrected sentence> Ebenso uralt sind ihre Träume einer erfolgreichen Karriere , in der sie ebenso viel Geld wie Männer verdienen könnten .</corrected sentence>\n<erroneous sentence> Es ist sinnlos eine ganze Sprache zu ändern , aber es ist wichtig zu merken wie männlich orientiert so eine ganztage Ding wie Kommunikation scheint .</erroneous sentence>\n<corrected sentence> Es ist sinnlos eine ganze Sprache zu ändern , aber es ist wichtig , zu merken , wie männlich-orientiert so ein ganztags Ding wie Kommunikation scheint .</corrected sentence>\n<erroneous sentence> Die Firmen wollen nicht Teorispezialisten mit hohen Noten haben !</erroneous sentence>\n<corrected sentence> Die Firmen wollen keine Theoriespezialisten mit hohen Noten haben !</corrected sentence>\n<erroneous sentence> Daher leistet man seinen Beitrag nicht für die Gesellschaft , sondern für sich selbst bzw. für seine Familie ;</erroneous sentence>\n<corrected sentence> Daher leistet man seinen Beitrag nicht für die Gesellschaft , sondern für sich selbst bzw. für seine Familie ;</corrected sentence>\n<erroneous sentence> Ich muss viel arbeiten , aber jetzt bin ich glücklich !</erroneous sentence>\n<corrected sentence> Ich muss viel arbeiten , aber jetzt bin ich glücklich !</corrected sentence>\n<erroneous sentence> Glücklicherweise funktionieren die Verkehrsmittel sehr gut , und sie sind billig .</erroneous sentence>\n<corrected sentence> Glücklicherweise funktionieren die Verkehrsmittel sehr gut , und sie sind billig .</corrected sentence>\n<erroneous sentence> Ohne genug Ärzte könnte eine große Bevölkerung nicht existieren , denn zu viele Menschen würden wegen Krankheit als Babys oder als Kinder sterben .</erroneous sentence>\n<corrected sentence> Ohne genug Ärzte könnte eine große Bevölkerung nicht existieren , denn zu viele Menschen würden wegen Krankheit als Babys oder als Kinder sterben .</corrected sentence>\n<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence> {target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'reproduce_space_ro_8fix': {
        'system': "",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\n{icl_examples}<erroneous sentence> Industria dronelor stă să explodeze în Uniunea Europeană în următori ani .</erroneous sentence>\n<corrected sentence> Industria dronelor stă să explodeze în Uniunea Europeană în următorii ani .</corrected sentence>\n<erroneous sentence> va depune o contestație la Curtea Constituțională exact pe acest proiect de lege</erroneous sentence>\n<corrected sentence> va depune o contestație la Curtea Constituțională exact în legătură cu acest proiect de lege</corrected sentence>\n<erroneous sentence> A spus că la sfârșitul lui august când sa întors în SUA imediat după incidentul din tren , își dorea doar să se relaxeze alături de familie .</erroneous sentence>\n<corrected sentence> A spus că la sfârșitul lui august , când s-a întors în SUA imediat după incidentul din tren , își dorea doar să se relaxeze alături de familie .</corrected sentence>\n<erroneous sentence> Ce se-ntâmplă cu oamenii aceia care rulează pe ecran?</erroneous sentence>\n<corrected sentence> Ce se-ntâmplă cu oamenii aceia ale căror poze rulează pe ecran?</corrected sentence>\n<erroneous sentence> Van Kleeck spune că Aerojet plănuia începerea testării motoarelor la scară largă în 2017 , urmată apoi de obținerea certificatului în 2019 , însă data ar putea fi amânată în cazul în care compania nu primește suficiente fonduri de la Forțele Aeriene în contracte așteptate la sfârșitul primului semestru al anului fiscal 2016 , care începe în octombrie .</erroneous sentence>\n<corrected sentence> Van Kleeck spune că Aerojet plănuia începerea testării motoarelor la scară largă în 2017 , urmată apoi de obținerea certificatului în 2019 , însă data ar putea fi amânată în cazul în care compania nu primește suficiente fonduri de la Forțele Aeriene în contracte așteptate la sfârșitul primului semestru al anului fiscal 2016 , care începe în octombrie .</corrected sentence>\n<erroneous sentence> Vă rog să vă puneți la punct întâi cu descrierea/licențierea și apoi cu limitele folosirii imaginilor sub clauza utilizării cinstite ( de obicei se admite una singură din acest tip într-un articol ) .</erroneous sentence>\n<corrected sentence> Vă rog să vă puneți la punct întâi cu descrierea/licențierea și apoi cu limitele folosirii imaginilor sub clauza utilizării cinstite ( de obicei se admite una singură din acest tip într-un articol ) .</corrected sentence>\n<erroneous sentence> În anul următor a invadat Portugalia , căpătând regiunea Algarve .</erroneous sentence>\n<corrected sentence> În anul următor a invadat Portugalia , căpătând regiunea Algarve .</corrected sentence>\n<erroneous sentence> A declarat că Bulgaria este o țară cu o istorie bogată , pe care studiile recente au plasat-o pe locul al treilea , după Italia și Grecia , în topul traficului de obiecte și bunuri culturale .</erroneous sentence>\n<corrected sentence> A declarat că Bulgaria este o țară cu o istorie bogată , pe care studiile recente au plasat-o pe locul al treilea , după Italia și Grecia , în topul traficului de obiecte și bunuri culturale .</corrected sentence>\n<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence> {target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'reproduce_space_et_8fix': {
        'system': "",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\n{icl_examples}<erroneous sentence> Isegi tudengid peavad nüüd enamus oma töödest esitama trükituna .</erroneous sentence>\n<corrected sentence> Isegi tudengid peavad nüüd enamuse oma töödest esitama trükituna .</corrected sentence>\n<erroneous sentence> Selles mõisas saab veel näha üle 300 erinevaid põõsaid .</erroneous sentence>\n<corrected sentence> Selles mõisas saab veel näha üle 300 erineva põõsa .</corrected sentence>\n<erroneous sentence> Selge on seda , et venelased ja eestlased vähe suhtlevad omavahel .</erroneous sentence>\n<corrected sentence> Selge on see , et venelased ja eestlased suhtlevad omavahel vähe .</corrected sentence>\n<erroneous sentence> Selleks , et soovitatud eesmärgi saavutada on vaja paljudest raskustest ja takistustest üle saada .</erroneous sentence>\n<corrected sentence> Selleks , et soovitud eesmärki saavutada , on vaja paljudest raskustest ja takistustest üle saada .</corrected sentence>\n<erroneous sentence> Ma tulin täna hommikul , nagu te teate , üsna vara siia ja teel siia nägin ma vähemalt kümmet meest , kes olid purjus ja räpased .</erroneous sentence>\n<corrected sentence> Ma tulin täna hommikul , nagu te teate , üsna vara siia ja teel siia nägin ma vähemalt kümmet meest , kes olid purjus ja räpased .</corrected sentence>\n<erroneous sentence> Aga elus on peaaegu kõik tasuline , me peame kõige eest maksma .</erroneous sentence>\n<corrected sentence> Aga elus on peaaegu kõik tasuline , me peame kõige eest maksma .</corrected sentence>\n<erroneous sentence> Tema väga hästi aru saanud sellest . et mida tähendab termin ' sociare ' ( lad keeles - kokku koguma ) .</erroneous sentence>\n<corrected sentence> Tema väga hästi aru saanud sellest . et mida tähendab termin ' sociare ' ( lad keeles - kokku koguma ) .</corrected sentence>\n<erroneous sentence> Mina arvan , et see on väga hea raamat selleks , et aru saada kuidas hääldata häälikuid ja sõnu .</erroneous sentence>\n<corrected sentence> Mina arvan , et see on väga hea raamat selleks , et aru saada kuidas hääldata häälikuid ja sõnu .</corrected sentence>\n<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence> {target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'min_edit_fewshot_space': {
        'system': "You are an language expert who is responsible for grammatical, lexical and orthographic error corrections given an input sentence. Your job is to fix grammatical mistakes, awkward phrases, spelling errors, etc. following standard written usage conventions, but your corrections must be conservative. Please keep the original sentence (words, phrases, and structure) as much as possible. The ultimate goal of this task is to make the given sentence sound natural to native speakers without making unnecessary changes. Corrections are not required when the sentence is already grammatical and sounds natural.",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\n{icl_examples}<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence> {target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'min_edit_fewshot_space_strict': {
        'system': "You are a conservative grammatical error correction expert. Correct only clear grammatical, spelling, punctuation, word-form, and word-choice errors. Keep the original words, meaning, order, and style whenever possible. Do not paraphrase. If the sentence is already acceptable, copy it exactly.",
        'prompt': "Correct the erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Return exactly one corrected sentence between `<corrected sentence>` and `</corrected sentence>`. Do not explain. Do not add comments. Do not write `(no correction needed)`. Do not continue with another `<erroneous sentence>` example after the answer.\n{icl_examples}<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence> {target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'min_edit_fewshot_space_hints': {
        'system': "You are a conservative grammatical error correction expert. Correct only clear grammatical, spelling, punctuation, word-form, and word-choice errors. Keep the original words, meaning, order, and style whenever possible. Do not paraphrase. If the sentence is already acceptable, copy it exactly.",
        'prompt': "Correct the erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Return exactly one corrected sentence between `<corrected sentence>` and `</corrected sentence>`. Do not explain. Do not add comments. Use the optional checks only as weak hints; ignore a hint if it is not a real error.\n{icl_examples}<erroneous sentence> {source}</erroneous sentence>\n{description}\n<corrected sentence>",
        'icl_example': "<erroneous sentence> {source}</erroneous sentence>\n<corrected sentence> {target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
}


class PromptTemplate:
    def __init__(self, template) -> None:
        assert template in ICL_TEMPLATES, f"{template} not supported."
        self.template = ICL_TEMPLATES[template]

    def get_answer_start(self):
        return self.template['answer_start']

    def get_answer_end(self):
        return self.template['answer_end']

    def format(self, **kwargs):
        final_instruction = self.template['prompt'].format(**kwargs)
        return self.template['system'], final_instruction

    def postprocess(self, output_str):
        if self.template["answer_start"] and self.template["answer_start"] in output_str:
            output_str = output_str.split(self.template["answer_start"], 1)[1]
        if self.template["answer_end"] and self.template["answer_end"] in output_str:
            output_str = output_str.split(self.template["answer_end"], 1)[0]
        return output_str


class ICLPromptTemplate(PromptTemplate):
    def __init__(self, template) -> None:
        assert template in ICL_TEMPLATES, f"{template} not supported."
        self.template = ICL_TEMPLATES[template]

    def _remove_trailing_answer_start(self, instruction):
        answer_start = self.template.get('answer_start', '')
        if not answer_start:
            return instruction

        stripped = instruction.rstrip()
        if not stripped.endswith(answer_start):
            return instruction

        return stripped[:-len(answer_start)].rstrip()

    def format(self, examples_list, **kwargs):
        icl_examples = ''
        for example in examples_list:
            icl_examples += self.template['icl_example'].format(**example)
        kwargs['icl_examples'] = icl_examples
        final_instruction = self.template['prompt'].format(**kwargs)
        prefill_answer_start = os.environ.get("GER_PREFILL_ANSWER_START", "").strip()
        if prefill_answer_start not in {"1", "true", "True", "TRUE", "yes", "Yes", "YES"}:
            final_instruction = self._remove_trailing_answer_start(final_instruction)
        return self.template['system'], final_instruction
