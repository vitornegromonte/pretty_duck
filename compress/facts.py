"""Templated factual-recall domains for circuit probes.

Each domain is a prompt template plus (subject, answer) pairs.  A minimal pair
swaps only the subject, so prompts differ in the subject tokens and nowhere
else; the model's answer is the first token of `` <answer>``.  Facts the model
does not know are filtered out later by the probe (``require_correct``), so the
lists only need to be *true*, not guaranteed to be known.  Domains use
different subject vocabularies on purpose: they are the "different knowledge"
whose circuits we check for entanglement.
"""

from __future__ import annotations

# Few-shot prefixes (constant across prompts, subjects not in the fact lists) make
# the answer format unambiguous: the next token is the answer, not a quote or colon.
TEMPLATES = {
    "capitals": "The capital of {} is",
    "elements": ("The chemical symbol for hydrogen is H.\n"
                 "The chemical symbol for helium is He.\n"
                 "The chemical symbol for {} is"),
    "french": ("English: tomato\nFrench: tomate\n"
               "English: banana\nFrench: banane\n"
               "English: {}\nFrench:"),
}

CAPITALS = [
    ("France", "Paris"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"),
    ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Greece", "Athens"), ("Portugal", "Lisbon"),
    ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
    ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Hungary", "Budapest"), ("Ireland", "Dublin"),
    ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Cuba", "Havana"), ("Kenya", "Nairobi"), ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"), ("Pakistan", "Islamabad"),
    ("Afghanistan", "Kabul"), ("Nepal", "Kathmandu"), ("Syria", "Damascus"), ("Jordan", "Amman"),
    ("Lebanon", "Beirut"), ("Venezuela", "Caracas"), ("Colombia", "Bogota"), ("Ecuador", "Quito"),
    ("Uruguay", "Montevideo"), ("Morocco", "Rabat"), ("Algeria", "Algiers"), ("Tunisia", "Tunis"),
    ("Libya", "Tripoli"), ("Sudan", "Khartoum"), ("Ghana", "Accra"), ("Senegal", "Dakar"),
    ("Angola", "Luanda"), ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"), ("Uganda", "Kampala"),
    ("Iceland", "Reykjavik"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"),
    ("Switzerland", "Bern"), ("Czechia", "Prague"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
    ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Latvia", "Riga"), ("Estonia", "Tallinn"),
    ("Lithuania", "Vilnius"), ("Bangladesh", "Dhaka"), ("Qatar", "Doha"), ("Oman", "Muscat"),
    ("Yemen", "Sanaa"), ("Cyprus", "Nicosia"), ("Malta", "Valletta"), ("Albania", "Tirana"),
    ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"), ("Uzbekistan", "Tashkent"),
    ("Jamaica", "Kingston"), ("Nigeria", "Abuja"), ("Somalia", "Mogadishu"), ("Mali", "Bamako"),
    ("Niger", "Niamey"), ("Rwanda", "Kigali"), ("Mozambique", "Maputo"), ("Namibia", "Windhoek"),
    ("Botswana", "Gaborone"), ("Madagascar", "Antananarivo"), ("Liberia", "Monrovia"),
    ("Eritrea", "Asmara"), ("Mauritania", "Nouakchott"),
]

ELEMENTS = [
    ("lithium", "Li"), ("carbon", "C"), ("nitrogen", "N"),
    ("oxygen", "O"), ("fluorine", "F"), ("neon", "Ne"), ("sodium", "Na"), ("magnesium", "Mg"),
    ("aluminum", "Al"), ("silicon", "Si"), ("phosphorus", "P"), ("sulfur", "S"), ("chlorine", "Cl"),
    ("argon", "Ar"), ("potassium", "K"), ("calcium", "Ca"), ("iron", "Fe"), ("copper", "Cu"),
    ("zinc", "Zn"), ("silver", "Ag"), ("tin", "Sn"), ("gold", "Au"), ("mercury", "Hg"),
    ("lead", "Pb"), ("platinum", "Pt"), ("nickel", "Ni"), ("cobalt", "Co"), ("uranium", "U"),
    ("titanium", "Ti"), ("chromium", "Cr"), ("manganese", "Mn"), ("boron", "B"), ("bromine", "Br"),
    ("iodine", "I"), ("krypton", "Kr"), ("xenon", "Xe"), ("radon", "Rn"), ("tungsten", "W"),
    ("barium", "Ba"), ("beryllium", "Be"), ("arsenic", "As"), ("cadmium", "Cd"), ("plutonium", "Pu"),
    ("gallium", "Ga"),
]

FRENCH = [
    ("dog", "chien"), ("cat", "chat"), ("house", "maison"), ("water", "eau"), ("bread", "pain"),
    ("apple", "pomme"), ("book", "livre"), ("car", "voiture"), ("friend", "ami"), ("sun", "soleil"),
    ("moon", "lune"), ("tree", "arbre"), ("flower", "fleur"), ("milk", "lait"), ("cheese", "fromage"),
    ("wine", "vin"), ("horse", "cheval"), ("bird", "oiseau"), ("fish", "poisson"), ("door", "porte"),
    ("street", "rue"), ("night", "nuit"), ("day", "jour"), ("red", "rouge"), ("blue", "bleu"),
    ("green", "vert"), ("white", "blanc"), ("black", "noir"), ("big", "grand"), ("small", "petit"),
    ("good", "bon"), ("king", "roi"), ("queen", "reine"), ("girl", "fille"), ("sea", "mer"),
    ("mountain", "montagne"), ("city", "ville"), ("hand", "main"), ("cold", "froid"), ("hot", "chaud"),
    ("bed", "lit"), ("chair", "chaise"), ("salt", "sel"), ("sugar", "sucre"), ("rain", "pluie"),
    ("snow", "neige"), ("wind", "vent"), ("fire", "feu"), ("earth", "terre"), ("time", "temps"),
    ("week", "semaine"), ("month", "mois"), ("winter", "hiver"), ("spring", "printemps"),
]

FACTS = {"capitals": CAPITALS, "elements": ELEMENTS, "french": FRENCH}


def prompt(domain: str, subject: str) -> str:
    return TEMPLATES[domain].format(subject)
