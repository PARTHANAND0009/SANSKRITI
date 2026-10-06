# Stem templates by question_type

21726 rows (stage 0 output). Coverage = share of the question type's rows.

## Association (5453 rows)

| rule | rows | coverage |
|---|---|---|
| assoc_regions_home_to | 1607 | 29.5% |
| assoc_where_famous | 1607 | 29.5% |
| assoc_or_country_associated_to | 1607 | 29.5% |
| none | 475 | 8.7% |
| lexicon | 99 | 1.8% |
| quoted | 58 | 1.1% |

**assoc_regions_home_to**  `^Which of the given regions is home to the (?P<E>.+?)\s*\?$`
- Which of the given regions is home to the Gamosa?  →  entity: `Gamosa` (stem)
- Which of the given regions is home to the Mirasol Lake Garden?  →  entity: `Mirasol Lake Garden` (stem)
- Which of the given regions is home to the Bhadohi carpet weaving art?  →  entity: `Bhadohi carpet weaving art` (stem)

**assoc_where_famous**  `^Where is the (?P<E>.+?) famous(?: within (?P<S>.+?))?\s*\?$`
- Where is the Mekhela Sador famous within Assam?  →  entity: `Mekhela Sador` (stem)
- Where is the Zampa Gateway famous within Dadra_and_Nagar_Haveli_and_Daman_and_Diu?  →  entity: `Zampa Gateway` (stem)
- Where is the Glassware famous within Uttar_Pradesh?  →  entity: `Glassware` (stem)

**assoc_or_country_associated_to**  `^(?:The )?(?P<E>.+?) is associated to which (?:region|country|state)(?: of (?P<S>.+?))?\s*\?$`
- Muga silk is associated to which region of Assam?  →  entity: `Muga silk` (stem)
- Devka Beach is associated to which region of Dadra_and_Nagar_Haveli_and_Daman_and_Diu?  →  entity: `Devka Beach` (stem)
- Brassware is associated to which region of Uttar_Pradesh?  →  entity: `Brassware` (stem)

**none**
- The influence of Tibetan Buddhism is prominently visible in the monasteries and lifestyle of the people in this state.  →  no entity
- Which stateâs Wancho tribe creates intricate wood carvings of headhunting scenes for Morung (youth dormitories)?  →  no entity
- The unique attire of this region includes women wearing Bhotu and a Nath, signifying their cultural pride.  →  no entity

**lexicon**
- Which Indian state's Adi tribe practices the Solung Festival, involving ritual sacrifices of mithun (bovine) and bamboo-based ceremonial structures?  →  entity: `Solung Festival` (stem)
- The tradition of Chaar Dham Yatra, which includes visiting Yamunotri and Gangotri, is primarily associated with this state.  →  entity: `Gangotri` (stem)
- Where do the Sentinelese people maintain complete isolation from outside contact, preserving pre-Neolithic traditions on North Sentinel Island?  →  entity: `Sentinelese` (stem)

**quoted**
- Which state is known as the "rice bowl of central India" with traditional farming practices celebrated during the Navakhana first rice harvest festival?  →  entity: `rice bowl of central India` (stem)
- The "Bastar Palace" showcases unique architecture and serves as a historical symbol for the tribal rulers of this state.  →  entity: `Bastar Palace` (stem)
- This region is known for its production of "Kosa" silk, which is woven by local artisans using traditional methods.  →  entity: `Kosa` (stem)

## Country Prediction (5563 rows)

| rule | rows | coverage |
|---|---|---|
| country_home_to | 1607 | 28.9% |
| assoc_or_country_associated_to | 1607 | 28.9% |
| country_famous_for | 1607 | 28.9% |
| none | 473 | 8.5% |
| quoted | 200 | 3.6% |
| lexicon | 69 | 1.2% |

**country_home_to**  `^Which country is the home to (?P<E>.+?)\s*\?$`
- Which country is the home to Muga silk?  →  entity: `Muga silk` (stem)
- Which country is the home to Dudhni Lake?  →  entity: `Dudhni Lake` (stem)
- Which country is the home to Sanjhi art?  →  entity: `Sanjhi art` (stem)

**assoc_or_country_associated_to**  `^(?:The )?(?P<E>.+?) is associated to which (?:region|country|state)(?: of (?P<S>.+?))?\s*\?$`
- The Muga weaving is associated to which country?  →  entity: `Muga weaving` (stem)
- The Diu Fort is associated to which country?  →  entity: `Diu Fort` (stem)
- The Zari work is associated to which country?  →  entity: `Zari work` (stem)

**country_famous_for**  `^Which country is famous for the (?P<E>.+?)\s*\?$`
- Which country is famous for the Bihu dance costumes?  →  entity: `Bihu dance costumes` (stem)
- Which country is famous for the Satmaliya Deer Park?  →  entity: `Satmaliya Deer Park` (stem)
- Which country is famous for the Khurja Pottery?  →  entity: `Khurja Pottery` (stem)

**none**
- The Thepla, a spiced flatbread made with fenugreek leaves, is traditionally eaten in which country?  →  no entity
- With a rich legacy of folk dances and music, this country celebrates multiple festivals throughout the year, particularly highlighting goddess worship.  →  no entity
- Which country has a significant Christian population due to historical Portuguese influence?  →  no entity

**quoted**
- The "Mawa Bati," a sweet delicacy stuffed with dry fruits, is famous in which country?  →  entity: `Mawa Bati` (stem)
- The "Maheshwari Sarees," known for their silk and cotton blend, originate from which country?  →  entity: `Maheshwari Sarees` (stem)
- The "Rani Durgavati," a warrior queen who fought against the Mughals, belonged to which country?  →  entity: `Rani Durgavati` (stem)

**lexicon**
- The Battle of Haldighati, fought between Maharana Pratap and the Mughal army, took place in which country?  →  entity: `Battle of Haldighati` (stem)
- The celebration of traditional music during harvest, embodied in dances like Dalkhai, is a notable feature of this country.  →  entity: `Dalkhai` (stem)
- The Thar Desert, one of the world's largest arid regions and home to diverse wildlife, is located in which country?  →  entity: `Thar Desert` (stem)

## General Awareness (5328 rows)

| rule | rows | coverage |
|---|---|---|
| none | 1779 | 33.4% |
| ga_which_one_belongs | 1607 | 30.2% |
| ga_closely_associated | 1606 | 30.1% |
| lexicon | 277 | 5.2% |
| quoted | 59 | 1.1% |

**none**
- This area historically experienced the influence of both Buddhist and Hindu dynasties in shaping its spiritual landscape.  →  no entity
- Community gatherings often revolve around traditional music and dance.  →  no entity
- Hosting large community meals to foster harmony and unity during sacred occasions.  →  no entity

**ga_which_one_belongs**  `^Which one belongs to (?P<L>.+?)\s*\?$`
- Which one belongs to Sualkuchi, Kamrup district, Assam?  →  entity: `Muga silk` (answer)
- Which one belongs to Silvassa, Dadra_and_Nagar_Haveli_and_Daman_and_Diu?  →  entity: `Nakshatra Garden` (answer)
- Which one belongs to Moradabad, Uttar_Pradesh?  →  entity: `Brassware` (answer)

**ga_closely_associated**  `^According to you, which of the following is closely associated to (?P<L>.+?)\s*\??$`
- According to you, which of the following is closely associated to BayavÃÂ¼ Hill,Kohima  of Nagaland?  →  entity: `Nagaland State Museum` (answer)
- According to you, which of the following is closely associated to Uttar_Pradesh of Uttar_Pradesh?  →  entity: `Jhoola` (answer)
- According to you, which of the following is closely associated to Tirupati district of Andhra_Pradesh?  →  entity: `Tirumala Venkateswara Temple Andhra` (answer)

**lexicon**
- The collapse of Golconda Sultanate led to a significant shift in control to which empire?  →  entity: `Golconda` (stem)
- The Falaknuma Palace is primarily known for its:  →  entity: `Falaknuma Palace` (stem)
- Which type of costume is typically worn in Bharatanatyam?  →  entity: `Bharatanatyam` (stem)

**quoted**
- The annual festival of "Tulip Festival" in Kashmir celebrates which aspect of local life?  →  entity: `Tulip Festival` (stem)
- The annual festival of "Vishu" in Kerala is celebrated to mark which event?  →  entity: `Vishu` (stem)
- In Uttar Pradesh, what is the significance of the "Ganga Aarti"?  →  entity: `Ganga Aarti` (stem)

## State Prediction (5382 rows)

| rule | rows | coverage |
|---|---|---|
| state_famous_for | 1610 | 29.9% |
| state_houses | 1606 | 29.8% |
| state_options_associated | 1606 | 29.8% |
| none | 452 | 8.4% |
| lexicon | 84 | 1.6% |
| quoted | 24 | 0.4% |

**state_famous_for**  `^Which state is famous for (?P<E>.+?)\s*\?$`
- Which state is famous for Mahabaleshwar?  →  entity: `Mahabaleshwar` (stem)
- Which state is famous for Dal-Bati-Churma?  →  entity: `Dal-Bati-Churma` (stem)
- Which state is famous for Do-drul Chorten?  →  entity: `Do-drul Chorten` (stem)

**state_houses**  `^According to you, which of the following states houses the (?P<E>.+?)\s*\?$`
- According to you, which of the following states houses the Ntangki National Park?  →  entity: `Ntangki National Park` (stem)
- According to you, which of the following states houses the Khichdi Mela?  →  entity: `Khichdi Mela` (stem)
- According to you, which of the following states houses the Borra Caves Andhra?  →  entity: `Borra Caves Andhra` (stem)

**state_options_associated**  `^Which of the states given in the options is associated to (?P<E>.+?)\s*\?$`
- Which of the states given in the options is associated to Nagaland Zoological Park?  →  entity: `Nagaland Zoological Park` (stem)
- Which of the states given in the options is associated to Ashoka's Pillar and the Dhamek Stupa?  →  entity: `Ashoka's Pillar and the Dhamek Stupa` (stem)
- Which of the states given in the options is associated to Undavalli cave architecture?  →  entity: `Undavalli cave architecture` (stem)

**none**
- Where was the Zamorin Kingdom, known for its naval power and flourishing trade with Arab and European merchants, located?  →  no entity
- Which state is home to the Siddi community, whose members speak a mix of Gujarati and African-influenced dialects?  →  no entity
- Maand, a classical form of folk singing that narrates tales of valor and romance, is a musical tradition of which Indian state?  →  no entity

**lexicon**
- Which Indian state is home to the Betla National Park, one of India's first tiger reserves?  →  entity: `Betla National Park` (stem)
- In which region can you find the famous Pangong Tso lake, known for its breathtaking beauty and changing water colors?  →  entity: `Pangong Tso` (stem)
- In which state is the Gotmar Mela, a ritualistic event where two villages throw stones at each other across the Jam River, observed?  →  entity: `Gotmar Mela` (stem)

**quoted**
- Where is the traditional folk dance called "Lava" performed, which is a part of the local cultural heritage?  →  entity: `Lava` (stem)
- In which region is the Dosmoche Festival, also known as the "Festival of Scapegoat," celebrated to drive away evil spirits?  →  entity: `Festival of Scapegoat` (stem)
- Which Indian state is known as the "most Baptist state in the world", with a large percentage of its population following Baptist Christianity?  →  entity: `most Baptist state in the world` (stem)
