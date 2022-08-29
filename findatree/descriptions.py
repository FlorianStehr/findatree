from typing import Dict, List
import re
import numpy as np

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


#%%
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


#%%
def species_groupby_families(
    family_patterns: List[str],
    ) -> Dict:

    families = {}
    for family_id, pattern in enumerate(family_patterns):

        family = {}

        family['family_name'] = pattern
        family['species_names'] = [name for id, name in species_id_to_name().items() if bool(re.search(pattern, name, re.IGNORECASE))]
        family['species_ids'] = [id for id, name in species_id_to_name().items() if bool(re.search(pattern, name, re.IGNORECASE))]

        families[family_id] = family
    
    return families


#%%
def species_id_to_family_id(
    species_ids: np.ndarray,
    families: Dict,
    ) -> np.ndarray:
    
    species_to_family_map = {}
    
    for family_id, family in families.items():
        family_species_ids = family['species_ids']
        
        for species_id in family_species_ids:
            species_to_family_map[species_id] = family_id

    
    family_ids = np.ones(len(species_ids), dtype=np.int16) * (-1)
    for species_id, family_id in species_to_family_map.items():
        family_ids[species_ids == species_id] = family_id
    
    return family_ids

#%%
def print_summary(
    tnrs: np.ndarray,
    species_ids: np.ndarray,
    family_ids: np.ndarray,
    families,
    ) -> None:

    assert len(tnrs) == len(species_ids) == len(family_ids), "All input arrays must have same shape (n,)"

    #################### Total and tnr
    tnrs_unique, tnrs_counts = np.unique(tnrs, return_counts=True)
    print(f"{'Total number of crowns':<30}: {len(tnrs)}")
    print(f"{'Mean number of crowns per tnr':<30}: {np.mean(tnrs_counts):.1f}")
    print('_'*50)


    #################### Species
    # Get unique species ids and counts
    species_ids_unique, counts = np.unique(species_ids, return_counts=True)

    # Sort from highest count to lowest count
    sort_idx= np.flip(np.argsort(counts))
    counts = counts[sort_idx]
    species_ids_unique = species_ids_unique[sort_idx]

    print()
    print(f"{'species_id':>10}| {'species_name':<30}| {'count'}")
    print('-'*50)
    for id, count in zip(species_ids_unique, counts):    
        print(f"{id:>10}| {species_id_to_name()[id]:<30}: {count}")
    
    print('_'*50)

    #################### Family
    # Remove family ids < 0
    family_ids = family_ids[family_ids >= 0]

    # Get unique family ids and counts
    family_ids_unique, counts = np.unique(family_ids, return_counts=True)

    # Sort from highest count to lowest count
    sort_idx= np.flip(np.argsort(counts))
    counts = counts[sort_idx]
    family_ids_unique = family_ids_unique[sort_idx]

    print()
    print(f"{'family_id':>10}| {'family_name':<30}| {'count':<5} | {'species_names'}")
    print('-'*50)
    for id, count in zip(family_ids_unique, counts):    
        print(f"{id:>10}| {families[id]['family_name']:<30}: {count:<5} | {families[id]['species_names']}")
    
    print('_'*50)
    
