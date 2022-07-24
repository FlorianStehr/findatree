from typing import Dict

def species_name_to_id() -> Dict:    
    """Map tree species names (str) to ids (int)

    Returns
    -------
    Dict
        Keys are german tree species names (str), values correpond to identifier id (int)
    """
    mapper ={
        'Gemeine Fichte': 118,
        'Kirsche': 36,
        'Omorikafichte': 119,
        'Wildbirne': 40,
        'Sitkafichte': 120,
        'Trauben- oder Stieleiche': 98,
        'Gemeine Kiefer': 134,
        'Traubeneiche': 48,
        'Strobe': 133,
        'Stieleiche': 51,
        'Schwarzkiefer': 129,
        'Zerreiche': 41,
        'Spirke': 135,
        'Roteiche': 53,
        'Zirbelkiefer': 123,
        'Bergulme': 70,
        'Bergkiefer': 128,
        'Flatterulme': 71,
        'Weißtanne': 100,
        'Elsbeere': 66,
        'Küstentanne': 103,
        'Mehlbeere': 63,
        'Edeltanne': 106,
        'Speierling': 65,
        'Nordmannstanne': 104,
        'Vogelbeere': 64,
        'Eibe': 137,
        'Gemeine Birke': 10,
        'Europäische Lärche': 116,
        'Moorbirke': 11,
        'Japanische Lärche': 117,
        'Walnuss': 26,
        'Douglasie': 136,
        'Schwarznuss': 25,
        'Lebensbaum': 138,
        'Edelkastanie': 15,
        'Sonstiges Nadelholz': 199,
        'Schwarzerle': 7,
        'Rotbuche': 20,
        'Grauerle': 8,
        'Hainbuche': 13,
        'Grünerle': 9,
        'Winterlinde': 68,
        'Aspe': 35,
        'Sommerlinde': 69,
        'Schwarzpappel': 34,
        'Gemeine Esche': 22,
        'Pappelhybrid': 33,
        'Bergahorn': 5,
        'Weide': 62,
        'Spitzahorn': 4,
        'Frühblühende Traubenkirsche': 38,
        'Feldahorn': 1,
        'Spätblühende Traubenkirsche': 39,
        'Robinie': 56,
        'Sonstiges Laubholz': 99,
    }

    return mapper

def species_id_to_name() -> Dict:
    """Map tree species ids (int) to names (str)

    Returns
    -------
    Dict
        Keys are identifier ids (int) values correspond to german tree species names (str).
    """
    mapper = species_name_to_id()
    mapper_reversed = dict([(val, key) for key, val in mapper.items()])
    
    return mapper_reversed