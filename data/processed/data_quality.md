# SANSKRITI data quality notes

Source: `13ari/Sanskriti` (revision `8d8523795491a1484834d5cb85ed898b04839077`), 21853 rows.
`qid` is the 0-based row position over the published splits, formatted `sk00000`.
Matching compares the answer field with each option after trimming whitespace and
lowercasing.

## 1. Answer matches none of the four options (127 rows)

These rows were dropped from our analysis.

| qid | state | attribute | stem | answer | options |
|---|---|---|---|---|---|
| sk05204 | Uttarakhand | Language | An integral part of the festivals in this region is the performance of local dances, often performed in Kumaoni and Garhwali. | Uttarakhand | A. Uttrakhand<br>B. Chhattisgarh<br>C. Maharashtra<br>D. Punjab |
| sk11262 | Andhra_Pradesh | Costume | Which Prakasam district is known for delicate Venkatagiri cotton sarees with silver zari work? | Venkatagiri | A. Bapatla<br>B. Ongole<br>C. Markapur<br>D. Guntur |
| sk12246 | Dadra_and_Nagar_Haveli_and_Daman_and_Diu | Art | What natural materials are typically used in Warli paintings? | Warli paintings | A. Natural colors and cow dung<br>B. Watercolors<br>C. Acrylic colors<br>D. Oil paints |
| sk12265 | Dadra_and_Nagar_Haveli_and_Daman_and_Diu | Festivals | Which of the following are famous annual festivals celebrated in Dadra and Nagar Haveli? | Tarpa festival and Vansda festival | A. Tarpa Festival<br>B. Vansda Festival<br>C. Both a and b<br>D. Neither a nor b |
| sk12286 | Dadra_and_Nagar_Haveli_and_Daman_and_Diu | Nightlife | What is the name of the rooftop bar in Daman known for its groovy music? | Rooftop bar | A. Manpasand<br>B. Daman Culture<br>C. Bluesky<br>D. Poison |
| sk12288 | Dadra_and_Nagar_Haveli_and_Daman_and_Diu | Personalities | Ardeshar Faramji Khabardar, also known by his pen name Adal, was born in ? | Ardeshar Faramji Khabardar | A. Daman<br>B. Diu<br>C. Dadra<br>D. Nagar Haveli |
| sk12289 | Dadra_and_Nagar_Haveli_and_Daman_and_Diu | Personalities | Who mentioned Sudhir Phadke and Prabhakar Kulkarni as key figures in the liberation of Dadra and Nagar Haveli during a 2019 parliamentary debate? | Sudhir Phadke and Prabhakar Kulkarni | A. Nirmala Sitharaman<br>B. Amit Shah<br>C. Rajnath Singh<br>D. Narendra Modi |
| sk12349 | Dadra_and_Nagar_Haveli_and_Daman_and_Diu | Tourism | What is the name of the famous artificial lake in Dadra and Nagar Haveli? | Dhudni Lake | A. Daman Ganga Reservoir<br>B. Dudhni Lake<br>C. Vansda Lake<br>D. Vansda Lake |
| sk12387 | Delhi | Cultural_Common_Sense | The term "Dilli chalo" (Let's go to Delhi) is often used in which context? | Dilli chalo | A. Tourism<br>B. Sports events<br>C. Religious pilgrimage<br>D. Political protests |
| sk12441 | Delhi | History | Who was the founder of the Slave Dynasty that ruled Delhi in the 13th century? | Slave Dynasty | A. Iltutmish<br>B. Qutub-ud-din Aibak<br>C. Razia Sultan<br>D. Balban |
| sk12491 | Delhi | Personalities | Which famous Bollywood actor, known as the "King of Bollywood," was born and raised in Delhi? | King of Bollywood | A. Salman Khan<br>B. Shah Rukh Khan<br>C. Aamir Khan<br>D. Amitabh Bachchan |
| sk12498 | Delhi | Rituals_and_Ceremonies | The ceremonial changing of guards takes place at which location? | Ceremonial changing of guards | A. Victoria Memorial<br>B. Mysore Palace<br>C. Gateway of IndiaMemorial<br>D. Rashtrapati Bhavan |
| sk12560 | Delhi | Tourism | Which city is home to the Rashtrapati Bhavan, the official residence of the President of India? | Rashtrapati Bhavan | A. Chennai<br>B. Mumbai (Maharashtra)<br>C. New Delhi<br>D. Kolkata |
| sk12566 | Delhi | Transport | Which section of Delhi Metro Phase-IV was inaugurated in January 2025? | Delhi Metro | A. Janakpuri-Krishna Park section<br>B. Rithala-Kundli section<br>C. Sahibabad-New Ashok Nagar section<br>D. Dwarka-Najafgarh section |
| sk13441 | Jharkhand | Cuisine | What is the primary ingredient used in preparing Dhuska? | Primary ingredient in Dhuska | A. Wheat flour<br>B. Rice and lentils<br>C. Corn flour<br>D. Maize |
| sk13446 | Jharkhand | Cuisine | During which festival is Dhuska most commonly prepared in tribal households? | Festival when Dhuska is commonly prepared | A. Diwali<br>B. Sohrai<br>C. Sarhul<br>D. Chhath Puja |
| sk13450 | Jharkhand | Cuisine | In which season is Rugra naturally available in Jharkhand? | Rugra naturally available season | A. Summer<br>B. Monsoon<br>C. Winter<br>D. Spring |
| sk13459 | Jharkhand | Cuisine | Dhuska, a deep-fried savory dish, is a popular traditional food of which Indian state? | Dhuska dish popular state | A. Chhattisgarh<br>B. Jharkhand<br>C. Odisha<br>D. Madhya Pradesh |
| sk13460 | Jharkhand | Cuisine | Rugra is a type of which food category commonly found in Jharkhand? | Rugra food category | A. Leafy Vegetable<br>B. Root Vegetable<br>C. Wild Mushroom<br>D. Spiced Lentil |
| sk13463 | Jharkhand | Cultural_Common_Sense | Pathalgadi is widely practiced in which districts of Jharkhand? | Pathalgadi districts in Jharkhand | A. Ranchi, Khunti, and Gumla<br>B. Bokaro, Dhanbad, and Giridih<br>C. Deoghar, Dumka, and Jamtara<br>D. Jamshedpur, Chaibasa, and Saraikela |
| sk13464 | Jharkhand | Cultural_Common_Sense | Panchi is a traditional attire worn by women of which tribal community in Jharkhand? | Panchi traditional attire tribal community | A. Santhal<br>B. Munda<br>C. Oraon<br>D. Ho |
| sk13468 | Jharkhand | Cultural_Common_Sense | What type of fabric is traditionally used to make Panchi in Jharkhand? | Fabric used for Panchi | A. Silk<br>B. Cotton<br>C. Wool<br>D. Polyster |
| sk13471 | Jharkhand | Cultural_Common_Sense | Pathalgadi is primarily associated with which tribal community in Jharkhand? | Pathalgadi associated tribal community | A. Santhal<br>B. Munda<br>C. Bhil<br>D. Gond |
| sk13492 | Jharkhand | Festivals | In which Indian state is the Kol Rebellion commemorated as a symbol of tribal resistance? | Kol Rebellion commemorated state | A. Chhattisgarh<br>B. Jharkhand<br>C. Odisha<br>D. Madhya Pradesh |
| sk13500 | Jharkhand | History | The Kol Rebellion was triggered by which British policy that allowed outsiders to settle on tribal land? | British policy triggering Kol Rebellion | A. Doctrine of Lapse<br>B. Permanent Settlement Act<br>C. Ryotwari System<br>D. Bengal Tenancy Act |
| sk13505 | Jharkhand | History | What does the word "Pathalgadi" literally translate to in English? | Pathalgadi literal meaning | A. Stone Inscription<br>B. Land Boundary<br>C. Tribal Oath<br>D. Sacred Ground |
| sk13512 | Jharkhand | History | What method did the Kols primarily use during their rebellion? | Kol Rebellion primary method | A. Guerrilla warfare in forests<br>B. Direct military confrontation<br>C. Passive resistance and non-cooperation<br>D. Seeking help from foreign rulers |
| sk13516 | Jharkhand | History | Pathalgadi is based on which constitutional provision that grants special rights to Scheduled Tribes in India? | Pathalgadi constitutional provision | A. Article 370<br>B. Article 244 (Sixth Schedule)<br>C. Article 243 (Panchayati Raj Extension to Scheduled Areas - PESA)<br>D. Article 21 |
| sk13518 | Jharkhand | History | The Kol Rebellion of 1831â1832 was an uprising by the Kol tribal community against which ruling power? | Kol Rebellion ruling power opposed | A. Mughal Empire<br>B. British East India Company<br>C. Maratha Empire<br>D. Portuguese Colonial Rule |
| sk13537 | Jharkhand | History | Apart from the Kol tribe, which other tribal communities supported the rebellion? | Tribal communities supporting Kol Rebellion | A. Santhal and Munda<br>B. Bhil and Gond<br>C. Khasi and Meitei<br>D. Toda and Kuruba |
| sk13538 | Jharkhand | History | Who was the prominent leader of the Kol Rebellion? | Prominent leader of Kol Rebellion | A. Budhu Bhagat<br>B. Tilka Manjhi<br>C. Sidho and Kanho Murmu<br>D. Birsa Munda |
| sk13541 | Jharkhand | History | In response to the Pathalgadi movement, the government of which state launched awareness campaigns and legal actions? | State government responding to Pathalgadi movement | A. West Bangal<br>B. Chhattisgarh<br>C. Jharkhand<br>D. Odisha |
| sk13542 | Jharkhand | Language | Traditionally, which script was used to write the Nagpuri language? | Traditional script for Nagpuri | A. Devanagari and Kaithi<br>B. Ol Chiki<br>C. Brahmi<br>D. Only Kaithi |
| sk13544 | Jharkhand | Language | Kurukh is primarily spoken by which tribal community in the following states? | Kurukh spoken by tribal community | A. Chhattisgarh<br>B. Jharkhand<br>C. West Bangal<br>D. Odisha |
| sk13546 | Jharkhand | Language | Kurukh is written in which script specifically developed for it in Jharkhand? | Kurukh script developed in Jharkhand | A. Maharashtra<br>B. Tolong Siki (Jharkhand)<br>C. West Bangal<br>D. Odisha |
| sk13580 | Karnataka | Art | Which of the following Kannada movies is based on Bhuta Kola and gained national recognition? | Kannada movie based on Bhuta Kola | A. Kantara<br>B. KGF<br>C. Ulidavaru Kandanthe<br>D. Rangitaranga |
| sk13584 | Karnataka | Art | The borders of Ilkal sarees are often woven using which traditional technique? | Ilkal saree border technique | A. Kasuti embroidery<br>B. Chikankari work<br>C. Kondi weaving<br>D. Block printing |
| sk13589 | Karnataka | Art | The Jumbo Savari, an iconic procession of Mysuru Dasara, features which sacred idol placed on a golden howdah? | Jumbo Savari features which idol | A. Lord Ganesha<br>B. Goddess Lakshmi<br>C. Goddess Chamundeshwari<br>D. Lord Shiva |
| sk13591 | Karnataka | Art | The signature 'Tope Tenku' design, commonly seen on Ilkal sarees, appears on which part of the saree? | Tope Tenku' design location | A. Blouse<br>B. Pallu<br>C. Entire saree<br>D. Border |
| sk13597 | Karnataka | Art | Exceptional stone The Ilkal saree is predominantly associated with which region of Karnataka?craftsmanship is a tradition that has been nurtured in this culturally rich region. | Ilkal saree region | A. North Karnataka<br>B. Coastal Karnataka<br>C. Malnad region<br>D. South Karnataka |
| sk13602 | Karnataka | Art | The Ilkal saree gets its name from which town in Karnataka, known for its traditional handloom industry? | Ilkal saree town | A. Ilkal<br>B. Hubballi<br>C. Channapatna<br>D. Mysuru |
| sk13614 | Karnataka | Cuisine | The name "Bisi Bele Bath" literally translates to what in Kannada? | Meaning of "Bisi Bele Bath" | A. Hot Lentil Rice<br>B. Spicy Mixed Rice<br>C. Hot Lentil Curry<br>D. Hot Dal Soup |
| sk13617 | Karnataka | Cuisine | What is the key spice blend used in Bisi Bele Bath? | Key spice blend in Bisi Bele Bath | A. Sambar Powder<br>B. Bisi Bele Bath Masala<br>C. Garam Masala<br>D. Rasam Powder |
| sk13619 | Karnataka | Cuisine | Bisi Bele Bath is believed to have originated from which Karnataka royal dynastyâs kitchen? | Royal dynasty associated with Bisi Bele Bath | A. Vijayanagara Empire<br>B. Chalukyas<br>C. Wodeyars of Mysore<br>D. Hoysala Empire |
| sk13627 | Karnataka | Cuisine | Which lentil is the primary ingredient in Bisi Bele Bath? | Primary lentil in Bisi Bele Bath | A. Chana Dal<br>B. Toor Dal<br>C. Moong Dal<br>D. Urad Dal |
| sk13632 | Karnataka | Cultural_Common_Sense | Which modern-day Indian state has the most visible remnants of the Vijayanagara Empire? | Modern-day state with Vijayanagara remnants | A. Jummu Kashmir<br>B. Tamil Nadu<br>C. Karnataka<br>D. Uttar Pradesh |
| sk13634 | Karnataka | Cultural_Common_Sense | What is the traditional opening ceremony of Mysuru Dasara called, which takes place atop Chamundi Hill? | Traditional opening of Mysuru Dasara | A. Navaratri Pooja<br>B. Vijayadashami Darshan<br>C. Khasagi Durbar<br>D. Dasara Inauguration |
| sk13635 | Karnataka | Cultural_Common_Sense | The Torchlight Parade, marking the grand finale of Mysuru Dasara, takes place at which venue? | Torchlight Parade venue | A. Mysore Palace<br>B. Chamundi Hill<br>C. Karanji Lake<br>D. Bannimantap Grounds |
| sk13644 | Karnataka | Cultural_Common_Sense | Mysuru Dasara is celebrated to honor which Hindu goddess? | Mysuru Dasara honors | A. Lakshmi<br>B. Saraswati<br>C. Durga (Chamundeshwari)<br>D. Parvati |
| sk13646 | Karnataka | Cultural_Common_Sense | The Ilkal saree weaving tradition has received the Geographical Indication (GI) tag for its uniqueness. Which Indian state holds this GI status for Ilkal sarees? | State with GI status for Ilkal sarees | A. Manipur<br>B. Kerala<br>C. Goa<br>D. Karnataka |
| sk13650 | Karnataka | Cultural_Common_Sense | The Ayudha Puja performed during Mysuru Dasara is traditionally associated with the worship of what? | Ayudha Puja in Mysuru Dasara worships | A. Weapons and tools<br>B. Books and scriptures<br>C. Gold and silver<br>D. Musical instruments |
| sk13651 | Karnataka | Cultural_Common_Sense | Which tree is worshipped during Mysuru Dasara as a symbol of victory? | Tree worshipped during Mysuru Dasara | A. Banyan Tree<br>B. Neem Tree<br>C. Ashoka Tree<br>D. Banni Tree |
| sk13653 | Karnataka | Cultural_Common_Sense | The famous Dasara Exhibition, organized every year as part of Mysuru Dasara, was first started by which Mysore ruler? | Dasara Exhibition first started by | A. Tipu Sultan<br>B. Krishnaraja Wadiyar III<br>C. Jayachamaraja Wadiyar<br>D. Chamaraja Wadiyar X |
| sk13655 | Karnataka | Cultural_Common_Sense | The Huttari festival is mainly observed in which region of Karnataka? | Huttari festival region | A. Coastal Karnataka<br>B. North Karnataka<br>C. Malnad region<br>D. Kodagu (Coorg) |
| sk13657 | Karnataka | Cultural_Common_Sense | Bhuta Kola is deeply associated with which ethnic community in Karnataka? | Ethnic community associated with Bhuta Kola | A. Kodavas<br>B. Vokkaligas<br>C. Tuluvas<br>D. Kurubas |
| sk13660 | Karnataka | Cultural_Common_Sense | What does the term âHuttariâ mean in the context of this festival? | Meaning of "Huttari" | A. First harvest<br>B. Sacred fire<br>C. Victory celebration<br>D. Traditional dance |
| sk13664 | Karnataka | Cultural_Common_Sense | Bhuta Kola is primarily practiced in which region of Karnataka? | Bhuta Kola primary region | A. Malnad region<br>B. North Karnataka<br>C. Coastal Karnataka (Tulunadu)<br>D. Hyderabad-Karnataka |
| sk13666 | Karnataka | Cultural_Common_Sense | Which art form is prominently showcased during Mysuru Dasara as part of the cultural performances? | Art form showcased during Mysuru Dasara | A. Bharatanatyam<br>B. Yakshagana<br>C. Kathakali<br>D. Lavani |
| sk13686 | Karnataka | Dance_and_Music | Which traditional Karnataka musical instrument is primarily used in Yakshagana, a classical dance drama from the region? | Yakshagana musical instrument | A. Maddale<br>B. Tabla<br>C. Dholak<br>D. Sitar |
| sk13708 | Karnataka | Festivals | Which region of Karnataka is most famous for the Yakshagana tradition? | Famous Yakshagana region | A. North Karnataka<br>B. Malnad and Coastal Karnataka<br>C. Hyderabad-Karnataka region<br>D. Southern Karnataka |
| sk13713 | Karnataka | Festivals | Most Yakshagana performances are based on stories from which two Hindu epics? | Hindu epics in Yakshagana stories | A. Ramayana and Mahabharata<br>B. Bhagavad Gita and Vedas<br>C. Rig Veda and Upanishads<br>D. Jataka Tales and Panchatantra |
| sk13718 | Karnataka | Festivals | Which of the following festivals is grandly celebrated in Mysuru, Karnataka? | Kannada Language | A. Pongal<br>B. Mysuru Dasara<br>C. Bihu<br>D. Navratri |
| sk13720 | Karnataka | Festivals | In which season is Bhuta Kola traditionally performed? | Bhuta Kola traditional season | A. Winter<br>B. Summer<br>C. Post-harvest period (December to July)<br>D. Monsoon |
| sk13723 | Karnataka | Festivals | In which city does the famous Kadalekai Parishe take place every year? | Kadalekai Parishe festival city | A. Mysuru<br>B. Bengaluru<br>C. Hubballi<br>D. Mangaluru |
| sk13729 | Karnataka | Festivals | The Kadalekai Parishe festival in Karnataka is believed to have started as a tribute to which divine figure to protect the groundnut crops? | Kadalekai Parishe tribute to | A. Lord Ganesha<br>B. Lord Basava (Nandi)<br>C. Goddess Lakshmi<br>D. Lord Venkateshwara |
| sk13733 | Karnataka | Festivals | Yakshagana is believed to have originated in which historical period in Karnataka? | Yakshagana origin period | A. Maurya Period<br>B. Chola Dynasty<br>C. Vijayanagara Empire<br>D. Hoysala Empire |
| sk13736 | Karnataka | Festivals | The Kadalekai Parishe festival is held in which month, marking the beginning of the groundnut harvest season in Karnataka? | Kadalekai Parishe month | A. January<br>B. November<br>C. May<br>D. April |
| sk13737 | Karnataka | Festivals | Which traditional weapon is commonly carried by Kodavas during the Huttari festival celebrations? | Kodava weapon during Huttari | A. Bow and Arrow<br>B. Spear and Shield<br>C. Club and Mace<br>D. Kodava sword (odi kathi) |
| sk13756 | Karnataka | Festivals | Which Karnataka district is best known for its grand celebrations of the Huttari festival? | Huttari festival district | A. Chikmagalur<br>B. Kodagu (Coorg)<br>C. Mysuru<br>D. Belagavi |
| sk13760 | Karnataka | Festivals | The Kadalekai Parishe festival, dedicated to the first groundnut harvest of the season, is celebrated near which famous landmark in Karnataka? | Kadalekai Parishe famous | A. Mysore Palace<br>B. Nandi Hills<br>C. Basavanagudi Bull Temple<br>D. Gol Gumbaz |
| sk13763 | Karnataka | Festivals | The Huttari festival is primarily associated with which agricultural crop? | Huttari festival crop | A. Ragi<br>B. Coffee<br>C. Paddy (Rice)<br>D. Sugarcane |
| sk13774 | Karnataka | Festivals | Which of the following is a primary musical instrument used in Yakshagana performances? | Primary Yakshagana musical instrument | A. Mridangam<br>B. Maddale<br>C. Shehnai<br>D. Sitar |
| sk13776 | Karnataka | Festivals | Ugadi marks the beginning of which traditional calendar followed in Karnataka? | Traditional calendar for Ugadi in Karnataka | A. Vikram Samvat<br>B. Saka Samvat<br>C. Tamil Panchangam<br>D. Chandramana Panchanga |
| sk13782 | Karnataka | History | The Vijayanagara rulers were known for patronizing which famous Hindu philosopher? | Hindu philosopher patronized by Vijayanagara rulers | A. Adi Shankaracharya<br>B. Vyasatirtha<br>C. Ramanujacharya<br>D. Basaveshwara |
| sk13784 | Karnataka | History | The Wadiyars of Mysore continued the grand tradition of Mysuru Dasara after the fall of which empire? | Wadiyars continued Mysuru Dasara after fall of | A. Mughal Empire<br>B. Maratha Empire<br>C. Vijayanagara Empire<br>D. Chola Empire |
| sk13787 | Karnataka | History | In which Indian state do people primarily follow the religious and philosophical teachings of the Sharana Movement even today? | State following Sharana teachings today | A. Manipur<br>B. Kerala<br>C. Goa<br>D. Karnataka |
| sk13788 | Karnataka | History | The Vijayanagara Empire was founded in 1336 by which two brothers? | Founders of Vijayanagara Empire | A. Bukka Raya and Harihara Raya<br>B. Krishnadevaraya and Achyutaraya<br>C. Rajaraja Chola and Rajendra Chola<br>D. Tipu Sultan and Hyder Ali |
| sk13789 | Karnataka | History | The architectural style of Vijayanagara inspired which later South Indian dynasty? | South Indian dynasty influenced by Vijayanagara architecture | A. Cholas<br>B. Wodeyars of Mysore<br>C. Nayakas of Madurai<br>D. Travancore kings |
| sk13790 | Karnataka | History | Mysuru Dasara was first celebrated under the rule of which dynasty? | Mysuru Dasara first celebrated under | A. Hoysala<br>B. Vijayanagara<br>C. Chalukya<br>D. Pallava |
| sk13794 | Karnataka | History | Which of the following Karnataka rulers was a contemporary of Basavanna? | Contemporary ruler of Basavanna | A. Krishnadevaraya<br>B. Vishnuvardhana<br>C. Kittur Rani Chennamma<br>D. Shivaji |
| sk13797 | Karnataka | History | What was the capital city of the Vijayanagara Empire, now a UNESCO World Heritage Site? | Capital city of Vijayanagara Empire | A. Badami<br>B. Mysore<br>C. Hampi<br>D. Bijapur |
| sk13801 | Karnataka | History | Which famous Karnataka city was the center of the Sharana Movement during the 12th century? | Sharana Movement center city | A. Mysuru<br>B. Kalyana (Basava Kalyana)<br>C. Udupi<br>D. Shivamogga |
| sk13804 | Karnataka | History | The Sharana Movement led to the rise of which religious sect in Karnataka? | Religious sect from Sharana Movement | A. Vaishnavism<br>B. Lingayatism<br>C. Shaivism<br>D. Buddhism |
| sk13812 | Karnataka | History | The Vijayanagara Empire was established as a response to the expansion of which medieval power? | Vijayanagara Empire response to | A. The Cholas<br>B. The Delhi Sultanate<br>C. The Marathas<br>D. The British East India Company |
| sk13815 | Karnataka | History | The spiritual and philosophical texts associated with the Sharana Movement in Karnataka are known as what? | Sharana Movement texts | A. Vachanas<br>B. Upanishads<br>C. Puranas<br>D. Dohas |
| sk13823 | Karnataka | Language | Kannada, the official language of Karnataka, is classified under which language family? | Dravidian Language Family | A. Dravidian<br>B. Indo-Aryan<br>C. Tibeto-Burman<br>D. Austroasiatic |
| sk13844 | Karnataka | Language | Kannada is derived from which ancient language family? | Kannada's ancient language family | A. Indo-Aryan<br>B. Dravidian<br>C. Sino-Tibetan<br>D. Austroasiatic |
| sk13850 | Karnataka | Personalities | Sharana Basava Jayanti commemorates the birth anniversary of which famous Kannada saint and philosopher? | Sharana Basava Jayanti commemorates | A. Purandara Dasa<br>B. Kanaka Dasa<br>C. Basavanna<br>D. Madhvacharya |
| sk13856 | Karnataka | Personalities | Which Kannada poet and playwright has significantly contributed to the documentation and evolution of Yakshagana? | Yakshagana poet/playwright | A. Kuvempu<br>B. B. V. Karanth<br>C. Girish Karnad<br>D. Masti Venkatesha Iyengar |
| sk13860 | Karnataka | Rituals_and_Ceremonies | During Bhuta Kola, the performer is believed to be possessed by which entity? | Entity possessing Bhuta Kola performer | A. Hindu gods like Vishnu or Shiva<br>B. Spirits (Bhutas) of deities or daivas<br>C. Sages and saints of Karnataka<br>D. Celestial beings from Puranic texts |
| sk13862 | Karnataka | Rituals_and_Ceremonies | The Kadalekai Parishe festival is believed to have started as an offering to which divine or mythological figure? | Kadalekai Parishe mythological offering | A. Goddess Durga<br>B. The Sacred Bull, Nandi<br>C. Lord Vishnu<br>D. Lord Rama |
| sk13864 | Karnataka | Rituals_and_Ceremonies | On Sharana Basava Jayanti, which special dish is often prepared and distributed as Prasada? | Prasada on Sharana Basava Jayanti | A. Ragi Mudde<br>B. Kadubu<br>C. Payasa<br>D. Obbattu |
| sk13882 | Karnataka | Rituals_and_Ceremonies | Which instruments are predominantly used during Bhuta Kola performances? | Instruments in Bhuta Kola | A. Veena and Mridangam<br>B. Nadaswaram and Thavil<br>C. Chenda and Maddale<br>D. Drum (Dolu) and Cymbals (Taal) |
| sk13886 | Karnataka | Rituals_and_Ceremonies | Which of the following is a common ritual performed on Sharana Basava Jayanti? | Common ritual on Sharana Basava Jayanti | A. Chanting of Vachanas (poetic teachings of Basavanna)<br>B. Fire-walking festival<br>C. Kite flying festival<br>D. Procession of Yakshagana performers |
| sk14021 | Ladakh | Cuisine | What is the traditional Ladakhi bread called? | Traditional Bread | A. Kulcha<br>B. Khambir<br>C. Roti<br>D. Naan |
| sk14022 | Ladakh | Dance_and_Music | In which of the following regions is the Sgra-snyan NOT commonly played? | Sgra-snyan | A. Ladakh<br>B. Bhutan<br>C. Kerala<br>D. Tibet |
| sk14025 | Ladakh | Dance_and_Music | In which religious context is the Sgra-snyan often used | Sgra-snyan | A. Christian masses<br>B. Hindu festivals<br>C. Tibetan Buddhist festivals<br>D. Islamic ceremonies |
| sk14026 | Ladakh | Dance_and_Music | How many strings does a traditional Sgra-snyan typically have? | Sgra-snyan | A. 4<br>B. 5<br>C. 6 or 7<br>D. 10 |
| sk14028 | Ladakh | Dance_and_Music | Which of the following regions is known for using the Dramyin or Dranyen, a traditional lute-like instrument, in its folk music? | Dramyin or Dranyen | A. Assam<br>B. Rajasthan<br>C. Ladakh<br>D. Himachal Pradesh |
| sk14043 | Ladakh | Personalities | From which Union Territory of India was Colonel Chewang Rinchen? | Colonel Chewang Rinchen | A. Himachal Pradesh<br>B. Uttarakhand<br>C. Jammu and Kashmir<br>D. Ladakh |
| sk14051 | Ladakh | Personalities | How many Indian service personnel, including Colonel Chewang Rinchen, have been awarded the Maha Vir Chakra twice | Maha Vir Chakra | A. 4<br>B. 5<br>C. 6<br>D. 7 |
| sk14062 | Ladakh | Religion | What is the main religion practiced in Ladakh? | Religion: Buddhism | A. Hinduism<br>B. Islam<br>C. Christianity<br>D. Buddhism |
| sk14079 | Lakshadweep | Art | In which year did the Portuguese arrive in Lakshadweep? | Portuguese(map) | A. 1510<br>B. 1498<br>C. 1505<br>D. 1500 |
| sk14081 | Lakshadweep | Costume | What is the traditional attire worn by Lakshadweep women called? | Kachi Koli | A. Lehenga<br>B. Kachi<br>C. Salwar Kameez<br>D. Sari |
| sk14096 | Lakshadweep | History | In which year did the British introduce the Lakshadweep Regulation, which diminished the judicial authority of local chieftains? | Lakshadweep Regulation | A. 1915<br>B. 1910<br>C. 1912<br>D. 1908 |
| sk14098 | Lakshadweep | History | Which South Indian dynasty incorporated Lakshadweep into its expanding territory in the early 11th century? | Chola Dynasty | A. Pallavas<br>B. Cholas<br>C. Cheras<br>D. Pandyas |
| sk14099 | Lakshadweep | History | What was the old name of Lakshadweep before it was officially renamed in 1973? | old name of Lakshadweep | A. Amindivi Islands<br>B. Malabar Islands<br>C. Laccadive Islands<br>D. Coral Islands |
| sk14100 | Lakshadweep | History | Recent underwater explorations near which Lakshadweep island revealed the wreck of a possible 17th or 18th-century European warship? | European warship | A. Kavaratti<br>B. Agatti<br>C. Kalpeni<br>D. Bangaram |
| sk14101 | Lakshadweep | History | In which century did Muslim missionary activity begin in Lakshadweep, eventually leading to the conversion of all islanders to Islam? | Missionary activity | A. 5th century<br>B. 6th century<br>C. 7th century<br>D. 8th century |
| sk14103 | Lakshadweep | History | When was Lakshadweep constituted as a Union Territory of India? | Union Territory | A. 1947<br>B. 1950<br>C. 1973<br>D. 1956 |
| sk14113 | Lakshadweep | Tourism | In which Indian territory would you find the Bangaram Atoll, a popular tourist destination? | Bangaram Atoll | A. Lakshadweep<br>B. Rajasthan<br>C. Kerala<br>D. Tamil Nadu |
| sk14280 | Manipur | Cuisine | What is the name of the traditional Manipuri fermented fish used in many dishes? | Fish | A. Morok Metpa<br>B. Chakhao<br>C. Iromba<br>D. Ngari |
| sk14297 | Manipur | Dance_and_Music | What is the main theme of the Ras Lila dance? | Ras Lila dance | A. The Ramayana epic<br>B. Tribal folklore<br>C. The love story of Radha and Krishna<br>D. The life of Buddha |
| sk14311 | Manipur | Festivals | Which festival celebrated in Manipur is also known as Ningol Chakouba? | Ningol Chachaba | A. Gangai<br>B. Eid ul Fitr<br>C. Ningol Chakouba<br>D. Chiroba |
| sk14312 | Manipur | Festivals | The festival Lai Haraoba celebrates which aspect of Manipuri culture? | Lai Haraoba | A. Victory in wars<br>B. Worship of local deities (Umang Lai)<br>C. Marriage ceremonies<br>D. Harvesting rituals |
| sk14332 | Manipur | Festivals | What is the name of the Manipuri festival similar to Holi? | Kuvi Yaoshang | A. Lui Ngai Ni<br>B. Ningol Chakouba<br>C. Lai Haraoba<br>D. Yaoshang |
| sk14389 | Manipur | Tourism | Which national park in Manipur is known for its natural beauty? | Keibul Lamjao National Gardens | A. Keibul Lamjo National Gardens<br>B. Sendra<br>C. Morah<br>D. Duko Valley |
| sk15070 | Sikkim | Art | Which craft items in Sikkim, made from these materials are popular among rural folks? | Cane, Bamboo | A. Cane,Bamboo<br>B. Ceramic Tiles<br>C. Stone Idols<br>D. Paper Crafts |
| sk15230 | Sikkim | Tourism | Which revered monastery, 24 km from Gangtok, belongs to the BlackHat sect of Buddhism? | Rumtek Monastery | A. RumtekMonastery<br>B. EncheyMonastery<br>C. TashidingMonastery<br>D. PemayangtseMonastery |
| sk15396 | Tamil_Nadu | Rituals_and_Ceremonies | What is the historical significance of the Chola dynasty in Tamil Nadu? | They were known for their skill in metalworking and bronze sculptures. | A. They built the Taj Mahal<br>B. They were known for their skill in metalworking and bronze sculptures<br>C. They unified India under one rule<br>D. They introduced Gandhi to Indian politics |
| sk16266 | Uttarakhand | Tourism | When people hike the trails near the Ganges River, what cultural aspect do they often experience? | Spiritual messaging through rituals and ceremonies. | A. Urban nightlife<br>B. Historic events related to independence<br>C. Spiritual messaging through rituals and ceremonies<br>D. Entertainment and gaming culture |
| sk16276 | Uttarakhand | Tourism | What cultural practice is associated with visiting sacred sites like Gangotri and Yamunotri in India? | Coming for spiritual cleansing through rituals. | A. Placing a flag on the summit<br>B. Performing yoga and meditation<br>C. Coming for spiritual cleansing through rituals<br>D. Participating in music and dance competitions |
| sk17419 | Dadra_and_Nagar_Haveli_and_Daman_and_Diu | History | Which Indian Union Territory is home to several historic Portuguese forts built during colonial rule, including those from the 16th century? | Daman and Diu. | A. Gujarat<br>B. Daman and Diu<br>C. Maharashtra<br>D. Punjab |
| sk20317 | Rajasthan | Cuisine | Which state is renowned for its traditional dish Dal-Bati-Churma, which includes lentils, baked wheat balls, and a sweet dessert, especially during festive occasions? | Rajasthan | A. Rajasthani<br>B. Gujarati<br>C. Kashmiri<br>D. Bengali |
| sk21769 | West_Bengal | Cuisine | According to you, which of the following states houses the Machher Jhol? | West_Bengal | A. Andhra_Pradesh<br>B. Punjab<br>C. Andaman_and_Nicobar<br>D. Sikkim |
| sk21772 | West_Bengal | Cuisine | Which state is famous for Machher Jhol? | West_Bengal | A. Odisha<br>B. Chandigarh<br>C. Andaman_and_Nicobar<br>D. Meghalaya |
| sk21776 | West_Bengal | Cuisine | Which of the states given in the options is associated to Machher Jhol? | West_Bengal | A. Manipur<br>B. Andaman_and_Nicobar<br>C. Meghalaya<br>D. Puducherry |

## 2. Answer matches more than one option (10 rows)

The same text appears in two or more options, so the gold letter is not
determined. These rows are kept in our files but excluded from analysis.

| qid | state | attribute | stem | answer | options |
|---|---|---|---|---|---|
| sk01112 | Chhattisgarh | Art | Bamboo Basketry is associated to which region of Chhattisgarh? | Bastar district | A. Pahalgam<br>B. Birbhum district<br>C. Bastar District<br>D. Bastar district |
| sk13171 | Himachal_Pradesh | Festivals | Festivals in this culture often coincide with significant agricultural dates, showcasing local traditions with reverence. | Indian | A. Indian<br>B. Turkish<br>C. Indian<br>D. Finnish |
| sk15343 | Tamil_Nadu | Festivals | The Natyanjali Dance Festival is primarily a celebration of which art form? | Dance_and_Music | A. Dance_and_Music<br>B. Dance_and_Music<br>C. Painting<br>D. Theatre |
| sk18686 | Jharkhand | Tourism | Which Indian state is home to Rajrappa Temple, a famous religious site dedicated to Goddess Chinnamasta? | Jharkhand | A. Jharkhand<br>B. Jharkhand<br>C. Nagaland<br>D. Arunachal Pradesh |
| sk19231 | Kerala | Tourism | In which Indian state is Munnar, a popular hill station known for its tea plantations, located? | Kerala | A. Karnataka<br>B. Kerala<br>C. Kerala<br>D. Odisha |
| sk19361 | Lakshadweep | Tourism | Which Indian Union Territory is famous for its pristine coral atolls and lagoons? | Lakshadweep | A. Lakshadweep<br>B. Lakshadweep<br>C. West Bengal<br>D. Uttarakhand |
| sk19362 | Lakshadweep | Tourism | In which union territory of India is the island of Minicoy known for its unique cultural practices, including the traditional "Bodu" dance? | Lakshadweep | A. Lakshadweep<br>B. Kerala<br>C. Lakshadweep<br>D. Sikkim |
| sk19364 | Lakshadweep | Tourism | In which Indian Union Territory can tourists explore the rich marine life of the Arabian Sea? | Lakshadweep | A. Andaman Islands<br>B. Lakshadweep<br>C. Lakshadweep<br>D. Sikkim |
| sk20417 | Sikkim | Cuisine | Which Indian state is famous for its local alcoholic beverage Tongba, a fermented millet-based drink? | Sikkim | A. Uttarakhand<br>B. Sikkim<br>C. Sikkim<br>D. Goa |
| sk20566 | Sikkim | Tourism | In which Indian state is the famous Rumtek Monastery, an important center of Tibetan Buddhism, located? | Sikkim | A. Sikkim<br>B. Sikkim<br>C. Himachal Pradesh<br>D. Gujarat |

## 3. Two options identical, gold still unique (73 rows)

A distractor appears twice, so the question effectively has three distinct
choices. Kept and flagged (`duplicate_options`); not excluded.

| qid | state | attribute | stem | answer | options |
|---|---|---|---|---|---|
| sk01327 | Chhattisgarh | Rituals_and_Ceremonies | Which of the given regions is home to the Navakhana harvest ritual? | Bastar district | A. Birbhum district<br>B. Gangtok<br>C. Bastar district<br>D. Gangtok |
| sk14412 | Manipur | Tourism | What is the name of the valley famous for its scenic beauty in Manipur? | Dzukou Valley | A. Dzukou Valley<br>B. Siroi Village<br>C. Morah Valley<br>D. Morah Valley |
| sk21244 | Tripura | Cuisine | Which Indian stateâs tribal cuisine benefits from Omega 3-rich fish, commonly found in forest waterfalls? | Tripura | A. Odisha<br>B. Tripura<br>C. Assam<br>D. Odisha |
| sk21245 | Tripura | Cuisine | In which Indian state are dishes like Godok and Eggu prepared using turmeric leaves and small fish? | Tripura | A. Mizoram<br>B. Tripura<br>C. West Bengal<br>D. Mizoram |
| sk21246 | Tripura | Cuisine | Which Indian stateâs delicacies rely on fresh turmeric plants for cooking and worship in tribal households? | Tripura | A. Jharkhand<br>B. Tripura<br>C. Meghalaya<br>D. Jharkhand |
| sk21247 | Tripura | Cuisine | In which Indian state is the lush greenery conducive to raising fish species, including those used in Godok and Eggu? | Tripura | A. Nagaland<br>B. Tripura<br>C. Manipur<br>D. Nagaland |
| sk21248 | Tripura | Cuisine | Which Indian state incorporates the homegrown spice Kaching for added fragrance and flavor in local delicacies? | Tripura | A. Arunachal Pradesh<br>B. Tripura<br>C. Sikkim<br>D. Arunachal Pradesh |
| sk21251 | Tripura | Cuisine | Which Indian stateâs tribal cuisine includes fresh small fish captured during monsoons, reflecting a sustainable approach? | Tripura | A. Mizoram<br>B. Tripura<br>C. West Bengal<br>D. Mizoram |
| sk21256 | Tripura | Cuisine | Which Indian state is recognized for the Debbarma communityâs famous cooking style, featuring minimal spices? | Tripura | A. Meghalaya<br>B. Tripura<br>C. Jharkhand<br>D. Meghalaya |
| sk21257 | Tripura | Cuisine | In which Indian state do tribal communities collect small fish from forest streams during monsoons for year-round use? | Tripura | A. Manipur<br>B. Tripura<br>C. Nagaland<br>D. Manipur |
| sk21259 | Tripura | Cuisine | Which Indian stateâs Debbarma community is known for using the spice Kaching in fish, chicken, and pork recipes? | Tripura | A. Sikkim<br>B. Tripura<br>C. Arunachal Pradesh<br>D. Sikkim |
| sk21261 | Tripura | Cuisine | In which Indian state do tribal groups refrain from excessive spices, preserving the natural taste of local produce? | Tripura | A. West Bengal<br>B. Tripura<br>C. Mizoram<br>D. West Bengal |
| sk21265 | Tripura | Cultural_Common_Sense | In which Indian state are communities like the Tripuri historically influenced by the Bengali Hindusâ language and customs? | Tripura | A. Assam<br>B. Tripura<br>C. Odisha<br>D. Assam |
| sk21270 | Tripura | Cultural_Common_Sense | Which Indian state includes tribal communities like Uchoi and Mizo, who maintain rich folklores about demons and witches? | Tripura | A. West Bengal<br>B. Tripura<br>C. Mizoram<br>D. West Bengal |
| sk21272 | Tripura | Cultural_Common_Sense | Which Indian state features the palace Nilmahal and Ujjwantaâs palace library as prime historical attractions? | Tripura | A. Sikkim<br>B. Tripura<br>C. Arunachal Pradesh<br>D. Sikkim |
| sk21273 | Tripura | Cultural_Common_Sense | Which Indian state's culture comprises an array of myths and puzzles featuring goddess, demon, witch, and galaxy references? | Tripura | A. Odisha<br>B. Tripura<br>C. Assam<br>D. Odisha |
| sk21274 | Tripura | Cultural_Common_Sense | In which Indian state was the palace of Nilmahal constructed, reflecting historical grandeur for visitors and locals alike? | Tripura | A. Arunachal Pradesh<br>B. Tripura<br>C. Sikkim<br>D. Arunachal Pradesh |
| sk21279 | Tripura | Cultural_Common_Sense | Which Indian stateâs tribes, including the Rang and Dodge, maintain strong environmental ethics and cleanliness in local life? | Tripura | A. Sikkim<br>B. Tripura<br>C. Arunachal Pradesh<br>D. Sikkim |
| sk21282 | Tripura | Cultural_Common_Sense | Which Indian state has nearly 19 tribal communities, such as the Garo and Kuki, who continue to preserve forest traditions? | Tripura | A. West Bengal<br>B. Tripura<br>C. Mizoram<br>D. West Bengal |
| sk21284 | Tripura | Cultural_Common_Sense | In which Indian state do communities like Garo and Manipuri coexist alongside local tribes in dense forests? | Tripura | A. Manipur<br>B. Tripura<br>C. Nagaland<br>D. Manipur |
| sk21285 | Tripura | Cultural_Common_Sense | Which Indian state uses Sabroom and Chakra dialects of Bengali among the local tribal population? | Tripura | A. Nagaland<br>B. Tripura<br>C. Manipur<br>D. Nagaland |
| sk21295 | Tripura | Cultural_Common_Sense | Which Indian state's cultural lore includes references to the Milky Way galaxy as a path of death? | Tripura | A. Nagaland<br>B. Tripura<br>C. Manipur<br>D. Nagaland |
| sk21298 | Tripura | Cultural_Common_Sense | Which Indian stateâs culture recounts storms emanating from demon-inhabited realms as part of local mythology? | Tripura | A. Nagaland<br>B. Tripura<br>C. Manipur<br>D. Nagaland |
| sk21301 | Tripura | Cultural_Common_Sense | Which Indian state has cultural narratives that revolve around demons, witches, and gods, shaping local worldview? | Tripura | A. Mizoram<br>B. Tripura<br>C. West Bengal<br>D. Mizoram |
| sk21302 | Tripura | Cultural_Common_Sense | In which Indian state do locals perceive the galaxy in mythic terms, calling the Milky Way a passage to the afterlife? | Tripura | A. Odisha<br>B. Tripura<br>C. Assam<br>D. Odisha |
| sk21307 | Tripura | Cultural_Common_Sense | Which state has forest-dwelling tribes who trace their ancestry to regions between Tibet and present-day Northeast India? | Tripura | A. Arunachal Pradesh<br>B. Tripura<br>C. Sikkim<br>D. Arunachal Pradesh |
| sk21310 | Tripura | Cultural_Common_Sense | In which Indian state is the influence of Bengali Hindus notably seen due to extensive court usage of the Bengali language? | Tripura | A. Manipur<br>B. Tripura<br>C. Nagaland<br>D. Manipur |
| sk21313 | Tripura | Cultural_Common_Sense | Which Indian state recognizes 19 tribes, including the Rang and Dodge communities, that prefer living in forested areas? | Tripura | A. Meghalaya<br>B. Tripura<br>C. Jharkhand<br>D. Meghalaya |
| sk21315 | Tripura | Cultural_Common_Sense | In which Indian stateâs culture do the Tripuri, Rang, and Garo communities hold strong ties to forest-based livelihoods? | Tripura | A. Meghalaya<br>B. Tripura<br>C. Jharkhand<br>D. Meghalaya |
| sk21316 | Tripura | Cultural_Common_Sense | In which Indian state can visitors explore the famed Ujjwanta palace library to observe historically significant sculptures? | Tripura | A. Sikkim<br>B. Tripura<br>C. Arunachal Pradesh<br>D. Sikkim |
| sk21317 | Tripura | Cultural_Common_Sense | In which Indian state do tribes like Tripuri, Kuki, and Mizo predominantly inhabit forest regions? | Tripura | A. Assam<br>B. Tripura<br>C. Odisha<br>D. Assam |
| sk21318 | Tripura | Cultural_Common_Sense | Which Indian state's legends describe the rainbow as coming to earth for water, exemplifying the deep ecological beliefs? | Tripura | A. Jharkhand<br>B. Tripura<br>C. Meghalaya<br>D. Jharkhand |
| sk21320 | Tripura | Cultural_Common_Sense | Which Indian state has a rich tradition of folk stories describing witches, as well as flora and fauna from daily life? | Tripura | A. Assam<br>B. Tripura<br>C. Odisha<br>D. Assam |
| sk21321 | Tripura | Cultural_Common_Sense | Which Indian state is recognized for the cultural traditions of the Tripuri people, historically linked to the Tibetan region? | Tripura | A. Meghalaya<br>B. Tripura<br>C. Jharkhand<br>D. Meghalaya |
| sk21325 | Tripura | Cultural_Common_Sense | Which Indian state is historically known as 'very prosperous' for its folklore on gods, demons, and cosmic phenomena? | Tripura | A. Mizoram<br>B. Tripura<br>C. West Bengal<br>D. Mizoram |
| sk21327 | Tripura | Cultural_Common_Sense | In which Indian state did the Tripuri king historically adopt Bengali court language, influencing local culture? | Tripura | A. Mizoram<br>B. Tripura<br>C. West Bengal<br>D. Mizoram |
| sk21336 | Tripura | History | In which Indian state did the Tekipra State rule for centuries, though it was little known at the time? | Tripura | A. Jharkhand<br>B. Tripura<br>C. Meghalaya<br>D. Jharkhand |
| sk21338 | Tripura | History | Which Indian state had the old capital at Udaipur, changed in the 18th century, and then reestablished in Agartala? | Tripura | A. Jharkhand<br>B. Tripura<br>C. Meghalaya<br>D. Jharkhand |
| sk21340 | Tripura | History | Which Indian state is known as a peaceful territory since 2016, having resolved many historical conflicts? | Tripura | A. Nagaland<br>B. Tripura<br>C. Manipur<br>D. Nagaland |
| sk21341 | Tripura | History | In which Indian state did King Krishna Kishore Manikya reign between 1830-1850, as documented in the Rajmala? | Tripura | A. Meghalaya<br>B. Tripura<br>C. Jharkhand<br>D. Meghalaya |
| sk21342 | Tripura | History | Which Indian stateâs link between Kolkata and Agartala expanded from 350 km to 1700 km post the formation of East Pakistan? | Tripura | A. Odisha<br>B. Tripura<br>C. Assam<br>D. Odisha |
| sk21343 | Tripura | History | Which Indian state joined India in 1949, after the Queen of Tripura signed a formal agreement on 9 September of that year? | Tripura | A. West Bengal<br>B. Tripura<br>C. Mizoram<br>D. West Bengal |
| sk21347 | Tripura | History | In which Indian state did the 1971 war prompt reorganization of the northern borders, culminating in a new statehood status? | Tripura | A. Sikkim<br>B. Tripura<br>C. Arunachal Pradesh<br>D. Sikkim |
| sk21348 | Tripura | History | Which Indian state experienced tensions between local Tripuri communities and arriving Hindu Bengalis after 1949? | Tripura | A. Sikkim<br>B. Tripura<br>C. Arunachal Pradesh<br>D. Sikkim |
| sk21350 | Tripura | History | In which Indian state was the Tribal District Council established to protect indigenous peoples and mitigate violence? | Tripura | A. Manipur<br>B. Tripura<br>C. Nagaland<br>D. Manipur |
| sk21352 | Tripura | History | In which Indian state is the region historically mentioned in the Mahabharata and certain religious Puranas? | Tripura | A. Arunachal Pradesh<br>B. Tripura<br>C. Sikkim<br>D. Arunachal Pradesh |
| sk21353 | Tripura | History | Which Indian state gained full statehood on 21 January 1972, alongside Meghalaya and Manipur? | Tripura | A. Meghalaya<br>B. Tripura<br>C. Jharkhand<br>D. Meghalaya |
| sk21354 | Tripura | History | Which Indian state became a princely state under British rule, previously referred to as Tippera or Chakla Roshanabad? | Tripura | A. Sikkim<br>B. Tripura<br>C. Arunachal Pradesh<br>D. Sikkim |
| sk21355 | Tripura | History | In which Indian state were the Mughal rulers influential but never fully controlled the mountainous regions? | Tripura | A. Arunachal Pradesh<br>B. Tripura<br>C. Sikkim<br>D. Arunachal Pradesh |
| sk21356 | Tripura | History | Which Indian state was historically known as Tippera Hill, eventually adopting its modern name after reorganization? | Tripura | A. West Bengal<br>B. Tripura<br>C. Mizoram<br>D. West Bengal |
| sk21357 | Tripura | History | Which Indian stateâs border shift from 350 km to 1700 km to Kolkata was triggered by the creation of East Pakistan? | Tripura | A. Mizoram<br>B. Tripura<br>C. West Bengal<br>D. Mizoram |
| sk21358 | Tripura | History | Which Indian stateâs 15th-century 'Rajmala' text details the lineage of 179 kings, including Krishna Kishore Manikya? | Tripura | A. Odisha<br>B. Tripura<br>C. Assam<br>D. Odisha |
| sk21359 | Tripura | History | In November 1956, which Indian state was recognized as a Union Territory, before forming its own cabinet in July 1963? | Tripura | A. Jharkhand<br>B. Tripura<br>C. Meghalaya<br>D. Jharkhand |
| sk21361 | Tripura | History | In which Indian state did rulers adopt the British administrative style, enabling the formation of municipal corporations? | Tripura | A. Odisha<br>B. Tripura<br>C. Assam<br>D. Odisha |
| sk21363 | Tripura | History | In which Indian state did the boundaries shift over time, once bordering Sundarbans to the south and Kamarpura to the north? | Tripura | A. Assam<br>B. Tripura<br>C. Odisha<br>D. Assam |
| sk21364 | Tripura | History | Which Indian state was attacked by Muslim rulers from the 13th century, culminating in partial rule by 1733? | Tripura | A. Meghalaya<br>B. Tripura<br>C. Jharkhand<br>D. Meghalaya |
| sk21368 | Tripura | History | In which Indian state did Udaipur serve as the capital under King Krishna Manikya before shifting to Old Agartala? | Tripura | A. Manipur<br>B. Tripura<br>C. Nagaland<br>D. Manipur |
| sk21369 | Tripura | History | Which Indian state, once known as Kirat Desh, finds mention in Emperor Ashokaâs inscriptions? | Tripura | A. West Bengal<br>B. Tripura<br>C. Mizoram<br>D. West Bengal |
| sk21374 | Tripura | History | Which Indian state saw its capital move from Old Agartala to New Agartala in the 19th century under regal influence? | Tripura | A. Nagaland<br>B. Tripura<br>C. Manipur<br>D. Nagaland |
| sk21375 | Tripura | History | Which Indian state was designated a Part C state, eventually becoming a separate state on 21 January 1972? | Tripura | A. Assam<br>B. Tripura<br>C. Odisha<br>D. Assam |
| sk21376 | Tripura | History | Which Indian state was partly occupied by the Pakistan Army during the 1971 war, subsequently regaining land after the conflict? | Tripura | A. Assam<br>B. Tripura<br>C. Odisha<br>D. Assam |
| sk21378 | Tripura | History | Which Indian state's history is detailed in the 'Rajmala,' chronicling the King of Tripura and Muslim historians? | Tripura | A. Manipur<br>B. Tripura<br>C. Nagaland<br>D. Manipur |
| sk21381 | Tripura | History | In which Indian state did the Mughal Empire heavily influence the selection of local Tripuri kings? | Tripura | A. Mizoram<br>B. Tripura<br>C. West Bengal<br>D. Mizoram |
| sk21382 | Tripura | History | In which Indian state did a massive influx of Hindu Bengali refugees occur post-independence, altering the demographic landscape? | Tripura | A. Mizoram<br>B. Tripura<br>C. West Bengal<br>D. Mizoram |
| sk21383 | Tripura | History | Which Indian state's ruler, Bir Chandra Manikya, established the Agartala Municipal Corporation, inspired by British systems? | Tripura | A. Arunachal Pradesh<br>B. Tripura<br>C. Sikkim<br>D. Arunachal Pradesh |
| sk21385 | Tripura | History | In which Indian state were ancient rulers given the surname âFa,â a term meaning âfatherâ according to Rajmala? | Tripura | A. Nagaland<br>B. Tripura<br>C. Manipur<br>D. Nagaland |
| sk21386 | Tripura | Language | In which Indian state is Kokborok predominantly spoken alongside Bengali among tribal communities like Debbarma? | Tripura | A. Manipur<br>B. Tripura<br>C. Nagaland<br>D. Manipur |
| sk21387 | Tripura | Language | In which Indian state do the tribe inhabitants primarily converse in Kokborok while also adopting other regional tongues? | Tripura | A. Jharkhand<br>B. Tripura<br>C. Meghalaya<br>D. Jharkhand |
| sk21388 | Tripura | Language | Which Indian state fosters a multilingual identity, including languages such as Rankahal, Halam, and Bengali? | Tripura | A. Odisha<br>B. Tripura<br>C. Assam<br>D. Odisha |
| sk21389 | Tripura | Language | In which Indian state is Kokborok recognized as a key language, especially among communities like the Debbarma? | Tripura | A. Assam<br>B. Tripura<br>C. Odisha<br>D. Assam |
| sk21390 | Tripura | Language | In which Indian state do residents also speak Rankahal and Halam languages, reflecting cultural diversity? | Tripura | A. Arunachal Pradesh<br>B. Tripura<br>C. Sikkim<br>D. Arunachal Pradesh |
| sk21391 | Tripura | Language | Which Indian stateâs tribal populations speak a variety of Bengali dialects, including Sabroom, in daily communication? | Tripura | A. Meghalaya<br>B. Tripura<br>C. Jharkhand<br>D. Meghalaya |
| sk21392 | Tripura | Language | Which Indian state officially uses English for administrative purposes but relies on Kokborok in everyday life? | Tripura | A. West Bengal<br>B. Tripura<br>C. Mizoram<br>D. West Bengal |

## 4. Stem reveals the answer (1974 rows)

The gold option, or a name form / demonym of the gold state, appears in the
question text. Kept and flagged (`leaks_answer`); excluded from our depth
analysis by default. Full list: `leaks.csv`.

| question_type | gold_country_name | gold_state_name | gold_text_in_stem |
|---|---|---|---|
| Association | 0 | 56 | 1211 |
| Country Prediction | 17 | 0 | 47 |
| General Awareness | 0 | 7 | 61 |
| State Prediction | 0 | 304 | 271 |

## 5. Association questions whose answer is the state itself (1891 rows)

Association questions ask *which region* an item belongs to, and the distractors are
regions (districts, valleys, towns). In these rows the gold option is instead the name
of the row's own state, so the question can be answered at state level, and when the
stem also names the state (e.g. "Where is X famous within Nagaland?" with gold
"Nagaland") the answer is given away. We flag the latter as `leaks_answer`. In the
templated stems the same items recur in each of the three Association templates
(equal row counts below), so the issue sits with those items' answer field.

| stem template | rows | stem also names the state |
|---|---|---|
| free-form | 601 | 28 |
| assoc_or_country_associated_to | 430 | 430 |
| assoc_regions_home_to | 430 | 81 |
| assoc_where_famous | 430 | 430 |

Examples:

| qid | state | attribute | stem | gold | options |
|---|---|---|---|---|---|
| sk01490 | Delhi | Art | Which of the given regions is home to the Mughal Miniature Paintings? | Delhi | A. Chhindwara<br>B. Sualkuchi, Kamrup district<br>C. Mahavir Enclave<br>D. Delhi |
| sk01710 | Delhi | Tourism | Where is the National Zoological Park famous within Delhi? | Delhi | A. Bhopal<br>B. Madhavamala village in Yerpedu mandal of Chittoor district<br>C. Delhi<br>D. Varanasi |
| sk03102 | Ladakh | Tourism | Which of the given regions is home to the Sham Valley Trek? | Ladakh | A. Namchi district<br>B. Ladakh<br>C. Darbhanga<br>D. Manaskhand |

A related, smaller case: in 15 Association rows the gold option repeats the
item named in the stem (e.g. "Which of the given regions is home to the Nicobari
pig-farming customs?" with gold "Nicobari pig-farming customs").
