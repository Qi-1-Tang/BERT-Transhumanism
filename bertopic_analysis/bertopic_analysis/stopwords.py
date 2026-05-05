"""
停用词列表 — 供 topic_modeling.py 使用
根据实际220本书（20中国网文 + 200西方科幻/奇幻）的主角名和常见词汇

维护说明：
  - 每次发现新的噪音词，直接在对应分类下添加即可
  - 添加后重新运行 topic_modeling.py 生效
"""

CUSTOM_STOPWORDS = [
    # ================= 1. 中国网文 (20 Chinese Web Novels) =================

    # --- 一念永恒 (A Will Eternal) - Er Gen ---
    "bai", "xiaochun", "bai xiaochun", "du", "linglong",

    # --- 斗破苍穹 (Battle Through the Heavens) ---
    "xiao", "yan", "xiao yan", "yao", "xun",
    "nalan", "yanran", "medusa",

    # --- 盘龙 (Coiling Dragon) - I Eat Tomatoes ---
    "linley", "baruch", "bebe", "delia",

    # --- 修真聊天群 (Cultivation Chat Group) ---
    "song", "shuhang", "song shuhang",

    # --- 莽荒纪 (Desolate Era) - I Eat Tomatoes ---
    "ji", "ning", "ji ning",

    # --- 修真四万年 (Forty Millenniums of Cultivation) ---
    "li", "yao", "li yao", "ding", "lingdang",

    # --- 我欲封天 (I Shall Seal the Heavens) - Er Gen ---
    "meng", "hao", "meng hao", "xu", "qing",

    # --- 超神机械师 (The Legendary Mechanic) ---
    "han", "xiao", "han xiao",

    # --- 天道图书馆 (Library of Heaven's Path) ---
    "zhang", "xuan", "zhang xuan",

    # --- 诡秘之主 (Lord of the Mysteries) ---
    "klein", "moretti", "zhou", "mingrui",
    "sherlock", "moriarty", "gehrman", "sparrow",
    "fool", "tarot",

    # --- 怪物乐园 (Monster Paradise) ---
    "lin", "huang", "lin huang",

    # --- 口袋猎人维度 (Pocket Hunting Dimension) ---
    "lu", "ze", "lu ze",

    # --- 仙逆 (Renegade Immortal) - Er Gen ---
    "wang", "lin", "wang lin", "muwan",

    # --- 蛊真人 (Reverend Insanity) ---
    "fang", "yuan", "fang yuan",

    # --- 星辰变 (Stellar Transformations) - I Eat Tomatoes ---
    "qin", "yu", "qin yu", "hei",

    # --- 吞噬星空 (Swallowed Star) - I Eat Tomatoes ---
    "luo", "feng", "luo feng", "babata",

    # --- 妖神记 (Tales of Demons and Gods) ---
    "nie", "nie li",

    # --- 真武世界 (True Martial World) ---
    "yi", "yun", "yi yun", "xintong",

    # --- 全职法师 (Versatile Mage) ---
    "mo", "fan", "mo fan", "xinxia",

    # --- 巫界术士 (Warlock of the Magus World) ---
    "leylin", "farlier",


    # ================= 2. 西方科幻/奇幻 (200 Western Sci-Fi / Fantasy Novels) =================
    # 按系列/作者分组，然后按字母序列出独立小说

    # ------ A Song of Ice and Fire / ASOIAF (George R.R. Martin) ------
    # Game of Thrones, A Clash of Kings, A Storm of Swords, A Dance With Dragons
    "ned", "stark", "jon", "snow", "daenerys", "targaryen", "dany",
    "tyrion", "lannister", "cersei", "jaime", "arya", "sansa", "bran",
    "catelyn", "robb", "joffrey", "stannis", "davos", "brienne",
    "theon", "greyjoy", "littlefinger", "varys", "hodor", "samwell",
    "tarly", "melisandre", "tywin", "petyr", "baelish",
    "baratheon", "tyrell", "bolton", "martell", "tully",
    "winterfell", "westeros",

    # ------ Alvin Maker series (Orson Scott Card) ------
    # Seventh Son, Red Prophet, Prentice Alvin, Alvin Journeyman
    "alvin", "peggy", "taleswapper", "kumsaw", "tenskwa",
    "verily", "cooper",

    # ------ Ender's Game + Speaker for the Dead (Orson Scott Card) ------
    "wiggin", "graff", "bean",
    "novinha", "libo", "miro", "ela", "olhado",

    # ------ Broken Earth trilogy (N.K. Jemisin) ------
    # The Fifth Season, The Obelisk Gate, The Stone Sky
    "essun", "syenite", "damaya", "nassun", "hoa",
    "schaffa", "alabaster", "innon", "tonkee",

    # ------ Mars Trilogy + 2312 (Kim Stanley Robinson) ------
    # Red Mars, Green Mars, Blue Mars, 2312
    "boone", "maya", "toitovna", "chalmers", "nadia",
    "sax", "russell", "clayborne", "bogdanov",
    "hiroko", "nirgal", "randolph",
    "swan", "wahram", "genette",

    # ------ Vorkosigan Saga (Lois McMaster Bujold) ------
    # Falling Free, Barrayar, The Vor Game, Mirror Dance, Paladin of Souls
    "miles", "vorkosigan", "cordelia", "aral", "gregor",
    "ivan", "bothari", "bel", "thorne", "graf",
    "ista", "cattilara", "arhys", "foix",

    # ------ Connie Willis Time Travel ------
    # Doomsday Book, To Say Nothing of the Dog, Blackout, All Clear
    "kivrin", "engle", "dunworthy", "polly", "churchill",
    "eileen", "verity", "kindle", "schrapnell",

    # ------ Neil Gaiman ------
    # American Gods, Anansi Boys, The Ocean at the End of the Lane, The Graveyard Book
    "shadow", "wednesday", "czernobog",
    "charlie", "nancy", "spider", "anansi",
    "lettie", "hempstock",
    "bod", "owens", "silas", "scarlett",

    # ------ Tim Powers ------
    # The Anubis Gates, Last Call, Dinner at Deviant's Palace, Earthquake Weather
    "brendan", "doyle", "scott", "crane",
    "gregorio", "rivas", "cochran",

    # ------ Ursula K. Le Guin ------
    # The Left Hand of Darkness, The Dispossessed, Tehanu, Lavinia, Powers
    "genly", "estraven", "therem",
    "shevek", "takver",
    "tenar", "ged", "therru", "ogion",
    "lavinia", "aeneas", "turnus",
    "gavir", "sallo",

    # ------ China Mieville ------
    # The City and the City, The Scar, Iron Council
    "borlu", "tyador", "corwi", "dhatt",
    "bellis", "coldwine", "uther", "doul", "fennec", "tanner",
    "judah", "cutter", "ori",

    # ------ Gene Wolfe / Book of the New Sun ------
    # The Complete Book of the New Sun, The Claw of the Conciliator, Soldier of the Mist
    "severian", "thecla", "dorcas", "vodalus", "baldanders",
    "latro",

    # ------ Martha Wells ------
    # Network Effect, System Collapse, Witch King
    "mensah", "ratthi",
    "kai", "ziede", "tahren",

    # ------ Naomi Novik ------
    # Uprooted, Spinning Silver
    "agnieszka", "sarkan",
    "miryem", "wanda", "irina", "staryk",

    # ------ T. Kingfisher ------
    # A Sorceress Comes to Call, Nettle & Bone
    "hester", "marra", "fenris",

    # ------ Charlie Jane Anders ------
    # All the Birds in the Sky, The City in the Middle of the Night
    "patricia", "delfine", "laurence", "armstead",
    "sophie", "mouth",

    # ------ Rudy Rucker ------
    # Software, Wetware
    "cobb", "stahn", "mooney", "della", "taze",

    # ------ Joe Haldeman ------
    # The Forever War, Forever Peace, Camouflage
    "mandella", "marygay", "julian",

    # ------ Robert A. Heinlein ------
    # Starship Troopers, Double Star, Stranger in a Strange Land,
    # The Moon Is a Harsh Mistress, Job: A Comedy of Justice
    "johnny", "rico", "dubois",
    "lorenzo", "smythe", "bonforte",
    "jubal", "harshaw", "jill",
    "manuel", "manny", "wyoh",
    "hergensheimer", "margrethe",

    # ------ Vernor Vinge ------
    # A Fire Upon the Deep, A Deepness in the Sky, Rainbows End
    "ravna", "pham", "nuwen", "jefri", "johanna",
    "ezr", "vinh", "trixia", "bonsol", "qiwi",

    # ------ Arkady Martine ------
    # A Memory Called Empire, A Desolation Called Peace
    "mahit", "dzmare", "seagrass", "yskandr", "aghavn",
    "hibiscus", "antidote",

    # ------ Isaac Asimov ------
    # Foundation series, The Gods Themselves
    "hari", "seldon", "gaal", "dornick", "salvor", "hardin",
    "hober", "mallow", "mule", "bayta", "darell",
    "lamont", "hallam", "dua",

    # ------ Arthur C. Clarke ------
    # Rendezvous with Rama, The Fountains of Paradise
    "norton", "vannevar", "morgan", "rajasinghe",

    # ------ Philip K. Dick ------
    # The Man in the High Castle
    "tagomi", "juliana", "frink", "childan",

    # ------ Frank Herbert ------
    # Dune
    "paul", "atreides", "jessica", "stilgar", "chani",
    "baron", "harkonnen", "leto",

    # ------ Roger Zelazny ------
    # Lord of Light, This Immortal, Trumps of Doom
    "sam", "mahasamatman", "yama", "kali",
    "conrad", "nomikos",
    "merlin", "merle", "corey", "ghostwheel",

    # ------ Dan Simmons ------
    # Hyperion
    "kassad", "fedmahn", "shrike", "sol", "weintraub",
    "consul", "brawne", "lamia", "silenus",

    # ------ William Gibson ------
    # Neuromancer
    "case", "molly", "armitage", "wintermute",

    # ------ Richard K. Morgan ------
    # Altered Carbon
    "takeshi", "kovacs", "ortega",

    # ------ Individual Novels (alphabetical) ------

    # A Canticle for Leibowitz (Walter M. Miller)
    "francis", "zerchi", "taddeo",

    # A Master of Djinn (P. Djèlí Clark)
    "fatma", "siti", "hadia",

    # A Song for a New Day (Sarah Pinsker)
    "luce", "cannon", "rosemary",

    # A Time of Changes (Robert Silverberg)
    "kinnall", "darival",

    # Among Others (Jo Walton)
    "morwenna", "mori",

    # Ancillary Justice (Ann Leckie)
    "breq", "seivarden", "anaander", "mianaai",

    # Apex (Ramez Naam)
    "kade", "lane", "nakamura",

    # Babel (R.F. Kuang)
    "robin", "swift", "ramy", "victoire", "letitia", "griffin",

    # Babel-17 (Samuel R. Delany)
    "rydra", "wong",

    # Bannerless (Carrie Vaughn)
    "enid", "tomas", "dak",

    # Beauty (Sheri S. Tepper)
    "beauty", "carabosse", "grumpkin",

    # Bitter Angels (C.L. Anderson)
    "terese", "drajeske",

    # Brittle Innings (Michael Bishop)
    "danny", "boles", "jumbo", "clerval",

    # Case of Conscience (James Blish)
    "sanchez", "cleaver", "agronski",

    # Countdown City (Ben H. Winters)
    "hank", "palace", "milano",

    # Cyteen (C.J. Cherryh)
    "emory", "justin", "warrick",

    # Darwin's Radio (Greg Bear)
    "kaye", "lang", "rafelson", "dicken",

    # Dead Space (Kali Wallace) / The Healers' War (E.A. Scarborough)
    "marley", "kitty", "mcculley",

    # Diamond Age (Neal Stephenson)
    "nell", "hackworth", "miranda",

    # Downbelow Station (C.J. Cherryh)
    "signy", "mallory", "damon", "konstantin",

    # Dreamsnake (Vonda McIntyre)
    "snake", "arevin",

    # Elvissey (Jack Womack) / Emissaries from the Dead (Adam-Troy Castro)
    "isabel", "andrea", "cort",

    # Fahrenheit 451 (Ray Bradbury)
    "montag", "clarisse", "beatty", "mildred", "faber",

    # Flowers for Algernon (Daniel Keyes)
    "gordon", "kinnian", "algernon",

    # Four Hundred Billion Stars (Paul McAuley)
    "dorthy", "yoshida",

    # Gateway (Frederik Pohl)
    "robinette", "broadhead", "klara",

    # Harpist in the Wind (Patricia McKillip)
    "morgon", "raederle", "deth",

    # Harry Potter (J.K. Rowling) - Prisoner of Azkaban + Goblet of Fire
    "harry", "potter", "hermione", "ron", "weasley",
    "dumbledore", "snape", "hagrid", "sirius",
    "lupin", "voldemort", "cedric", "malfoy",

    # Headcrash (Bruce Bethke)
    "burroughs",

    # Hominids (Robert Sawyer)
    "ponter", "boddit", "adikor", "huld",

    # Homunculus (James P. Blaylock)
    "langdon", "ives", "narbondo", "owlesby",

    # Jonathan Strange & Mr Norrell (Susanna Clarke)
    "strange", "norrell", "childermass", "drawlight", "lascelles",

    # Kaiju Preservation Society (John Scalzi)
    "jamie", "gray",

    # King of Morning, Queen of Day (Ian McDonald)
    "caldwell", "enye", "maccoll",

    # Life (Gwyneth Jones)
    "senoz",

    # Lord Valentine's Castle (Robert Silverberg)
    "carabella", "hissune",

    # Lost Everything (Brian Francis Slattery)
    "sunny", "bauxite",

    # Making Money (Terry Pratchett)
    "moist", "lipwig", "vetinari", "adora",

    # Man Plus (Frederik Pohl)
    "torraway", "dorrie",

    # Middlegame (Seanan McGuire)
    "middleton", "dodger",

    # Moving Mars (Greg Bear)
    "casseia", "majumdar",

    # Mysterium (Robert Charles Wilson)
    "poole",

    # No Enemy But Time (Michael Bishop)
    "kampa",

    # Nova Swing (M. John Harrison)
    "vic", "serotonin", "aschemann",

    # Parable of the Talents (Octavia Butler)
    "olamina", "larkin", "bankole", "jarret",

    # Redshirts (John Scalzi)
    "dahl", "kerensky",

    # Ringworld (Larry Niven)
    "louis", "wu", "teela", "nessus",

    # Rite of Passage (Alexei Panshin)
    "havero",

    # Road Out of Winter (Alison Stine)
    "wylodine",

    # Seeker (Jack McDevitt)
    "kolpath", "benedict",

    # Ship of Fools (Richard Paul Russo)
    "bartolomeo",

    # Silmarillion (J.R.R. Tolkien)
    "feanor", "melkor", "morgoth", "beren", "luthien",
    "manwe", "fingolfin", "turin", "thingol",
    "elrond", "galadriel", "sauron",

    # Slow River (Nicola Griffith)
    "lore", "spanner",

    # Some Desperate Glory (Emily Tesh)
    "kyr", "valkyr", "mags",

    # Someone You Can Build a Nest In (John Wiswell)
    "shesheshen", "homily",

    # Spin (Robert Charles Wilson)
    "dupree", "lawton",

    # Spin Control (Chris Moriarty)
    "catherine",

    # Stand on Zanzibar (John Brunner)
    "niblock", "hogan", "mulligan",

    # Startide Rising (David Brin)
    "creideiki", "gillian", "baskin", "orley",

    # The Big Time (Fritz Leiber)
    "greta", "forzane",

    # The Book of the Unnamed Midwife (Meg Elison)
    # Unnamed protagonist - no character names to filter

    # The Calculating Stars (Mary Robinette Kowal)
    "elma", "york", "nathaniel",

    # The Demolished Man (Alfred Bester)
    "reich", "powell",

    # The Einstein Intersection (Samuel R. Delany)
    "lobey",

    # The Extractionist (Kimberly Unger)
    "brighton",

    # The Falling Woman (Pat Murphy)
    "diane",

    # The Goblin Emperor (Katherine Addison)
    "maia", "drazhar", "csethiro", "beshelar",

    # The Innkeeper's Song (Peter S. Beagle)
    "lal", "nyateneri", "lukassa", "soukyan",

    # The Man Who Saw Seconds (Alexander Boldizar)
    "kasper",

    # The Mists of Avalon (Marion Zimmer Bradley)
    "morgaine", "viviane", "gwenhwyfar", "lancelet",

    # The Moon and the Sun (Vonda McIntyre)
    "josèphe",

    # The Mount (Carol Emshwiller)
    "charley",

    # The Privilege of the Sword (Ellen Kushner)
    "vier",

    # The Quantum Rose (Catherine Asaro)
    "kamoj", "havyrl", "vyrl",

    # The Saint of Bright Doors (Vajra Chandrasekera)
    "fetter",

    # The Snow Queen (Joan Vinge)
    "dawntreader", "sparks", "arienrhod", "gundhalinu",

    # The Strange Affair of Spring Heeled Jack (Mark Hodder)
    "swinburne",

    # The Tainted Cup (Robert Jackson Bennett)
    "dinios", "kol", "dolabra",

    # The Terminal Experiment (Robert Sawyer)
    "hobson",

    # The Three-Body Problem (Liu Cixin)
    "ye", "wenjie", "miao",
    "cheng", "xin", "shi", "qiang",

    # The Time Ships (Stephen Baxter)
    "nebogipfel",

    # The Troika (Stepan Chapman)
    "naomi", "eva",

    # The Uplift War (David Brin)
    "oneagle", "athaclena", "fiben", "bolger",

    # The Wanderer (Fritz Leiber)
    "hagbolt", "merriam",

    # The Yiddish Policemen's Union (Michael Chabon)
    "landsman", "berko", "shemets",

    # Theory of Bastards (Audrey Schulman)
    "francine", "burk",

    # These Burning Stars (Bethany Jacobs)
    "jun", "ironway", "esek", "chono",

    # They'd Rather Be Right (Clifton & Riley)
    "bossy",

    # Through the Heart (Richard Grant)
    "kem",

    # Timescape (Gregory Benford)
    "bernstein", "renfrew",

    # To Your Scattered Bodies Go (Philip José Farmer)
    "hargreaves",

    # War Surf (M.M. Buckner)
    "nasir", "deepra",

    # Way Station (Clifford Simak)
    "enoch", "wallace",

    # Terminal Mind (David Walton)
    "darin", "kinsley",

    # Strange Toys (Patricia Geary)
    "pet",


    # ================= 3. 通用噪音 (General Noise) =================
    "chapter", "vol", "volume", "book", "books", "part",
    "prologue", "epilogue", "interlude",
    "translator", "editor", "translation", "notes",
    # --- 叙事动词 / 对话标签 (Narrative Verbs & Dialogue Tags) ---
    "said", "asked", "replied", "shouted", "whispered", "looked",
    "felt", "knew", "seemed", "told", "heard", "turned", "walked", "stood",
    "sat", "began", "nodded", "smiled", "laughed", "cried", "stared", "sighed",
    "pulled", "pushed", "moved", "ran", "went", "came", "got", "made", "took",
    "gave", "seen", "done", "called", "found", "kept", "tried", "left",
    "wanted", "needed", "put", "set",
    "see", "make", "take", "come", "go", "get", "let", "tell", "give",
    "find", "keep", "seem", "look", "feel", "try", "leave", "call", "turn",
    "want", "need", "know", "think", "stand",
    # --- 说话动词补充 (Speaking Verbs) ---
    "spoke", "muttered", "murmured", "exclaimed", "gasped", "groaned",
    "snapped", "stammered", "answered", "responded", "explained", "continued",
    "mentioned", "added", "repeated", "declared", "announced", "demanded",
    "insisted", "protested", "interrupted", "remarked", "suggested", "agreed",
    "pleaded", "urged", "warned",
    "screamed", "yelled", "roared", "sneered", "hissed", "growled",
    "wailed", "shrieked", "blurted", "scoffed", "smirked", "bellowed",
    "snickered", "scream", "yell", "roar", "sneer", "hiss", "growl",
    "mutter", "murmur", "exclaim", "gasp", "groan", "snap", "stammer",
    "answer", "respond", "explain", "mention", "repeat", "declare",
    "announce", "demand", "insist", "protest", "interrupt", "suggest",
    # --- 目视/观察动词 (Looking / Observing Verbs) ---
    "saw", "glanced", "peered", "gazed", "gaze", "gazes", "watched",
    "noticed", "observed", "spotted", "eyed", "squinted", "blinked",
    "stare", "glare", "glared", "peer", "glance", "watch",
    # --- 肢体/物理动作 (Physical Actions) ---
    "grabbed", "held", "reached", "touched", "gripped", "clutched",
    "squeezed", "pressed", "lifted", "raised", "dropped", "caught",
    "threw", "picked", "placed", "pointed", "waved", "gestured",
    "swung", "slammed", "kicked", "struck", "hit", "punched",
    "carry", "carried", "tossed", "handed", "hold",
    "reach", "touch", "grab", "throw", "catch", "pick", "lift",
    "raise", "drop", "press", "strike", "swing", "kick", "slam",
    # --- 肢体语言/生理反应 (Body Language & Reactions) ---
    "shook", "frowned", "shrugged", "winced", "flinched", "trembled",
    "shivered", "swallowed", "breathed", "exhaled", "inhaled", "snorted",
    "chuckled", "grinned", "grimaced", "scowled", "twitched", "clenched",
    "frown", "shrug", "tremble", "shiver", "wince", "flinch", "grin",
    "coughed", "spat", "wiped", "rubbed", "scratched",
    "narrowed", "widened", "furrowed", "flickered", "flashed",
    "cough", "spit", "wipe", "rub", "scratch", "widen",
    "shaking", "trembling", "sweating", "panting", "bleeding", "sobbing",
    # --- 移动动词 (Movement Verbs) ---
    "stepped", "rushed", "hurried", "dashed", "leaped", "jumped",
    "fell", "rose", "entered", "approached", "retreated", "emerged",
    "appeared", "disappeared", "returned", "passed", "crossed", "landed",
    "climbed", "descended", "stumbled", "charged", "sprinted", "crawled",
    "step", "rush", "hurry", "dash", "leap", "jump", "fall", "rise",
    "enter", "approach", "emerge", "return", "pass", "cross", "climb",
    "move", "appear", "disappear",
    # --- 心理/认知动词 (Mental Verbs) ---
    "realized", "wondered", "figured", "decided", "remembered",
    "recognized", "understood", "imagined", "considered", "supposed",
    "believed", "assumed", "hoped", "feared", "expected", "forgot",
    "realize", "wonder", "figure", "decide", "remember", "recognize",
    "understand", "imagine", "consider", "suppose", "believe", "assume",
    "hope", "fear", "expect", "forget",
    # --- 其他通用动作 (Other Generic Verbs) ---
    "echoed",
    "opened", "closed", "stopped", "paused", "started", "finished",
    "managed", "happened", "remained", "became", "changed", "grew",
    "brought", "sent", "lost", "broke", "wore", "hung",
    "led", "woke", "drew", "flew", "shone", "slid", "spun",
    "tore", "bent", "dug", "hid", "bound", "dealt", "sank", "swept",
    "open", "close", "stop", "pause", "start", "finish", "manage",
    "happen", "remain", "become", "change", "grow", "bring", "send",
    "lose", "break", "wear", "hang", "lead", "wake", "draw",
    # --- 感知/状态动词补充 (Perception / State Verbs) ---
    "hearing", "pondered", "filled", "sense",
    "ponder", "fill", "light",
    "sensed", "knowing", "looking", "looks", "causing", "considering",
    "surrounded", "startled", "rang", "ringing",
    "surrounding", "cause", "caused", "surround",
    # --- 高频 -ing 动名词 (Gerund / Present Participle — 极易泄漏) ---
    "feeling", "walking", "standing", "sitting", "running",
    "coming", "going", "saying", "telling", "getting", "making",
    "taking", "turning", "trying", "speaking", "holding", "reaching",
    "growing", "becoming", "following", "moving", "passing", "rising",
    "falling", "leaving", "opening", "closing", "remaining", "changing",
    "waiting", "watching", "listening", "pulling", "pushing", "carrying",
    "fighting", "working", "talking", "eating", "drinking", "sleeping",
    "flying", "floating", "glowing", "flowing", "burning", "spinning",
    "smiling", "laughing", "crying", "sighing", "nodding", "staring",
    "whispering", "shouting", "screaming", "yelling", "roaring",
    "stepping", "rushing", "jumping", "climbing", "crawling",
    "pointing", "waving", "touching", "pressing", "gripping",
    "frowning", "grinning", "sneering", "muttering", "murmuring",
    "appearing", "disappearing", "returning", "entering", "approaching",
    "wondering", "hoping", "expecting", "realizing", "noticing",
    "struggling", "managing", "beginning", "continuing", "stopping",

    # --- 不定代词 / -thing / -one / -where 系列 ---
    "something", "anything", "everything", "nothing",
    "someone", "anyone", "everyone", "no one", "nobody",
    "somewhere", "anywhere", "everywhere", "nowhere",
    "somehow", "anyway", "anyhow",

    # --- 副词 / 程度词 (Adverbs & Degree Words) ---
    "really", "quite", "rather", "almost", "nearly", "perhaps", "maybe",
    "always", "never", "often", "sometimes", "already", "still", "also",
    "even", "ever", "enough", "certainly", "probably", "apparently",
    "again", "away", "back",
    "suddenly", "surely", "involuntarily", "completely", "entire",
    "extremely", "incredibly", "colored",
    "hurriedly", "unexpectedly", "definitely", "shocking", "shockingly",
    "unexpected", "aged",
    # --- 叙事副词补充 (Narrative Adverbs) ---
    "immediately", "instantly", "quickly", "slowly", "finally", "eventually",
    "gradually", "slightly", "gently", "softly", "heavily", "deeply",
    "clearly", "simply", "merely", "barely", "hardly", "directly",
    "carefully", "silently", "quietly", "tightly", "lightly", "roughly",
    "firmly", "rapidly", "swiftly", "fiercely", "wildly", "desperately", "violently",
    "honestly", "seriously", "naturally", "obviously", "basically",
    "actually", "literally", "practically", "precisely", "exactly",
    "absolutely", "entirely", "utterly", "thoroughly", "partly", "fully",
    "mostly", "largely", "mainly", "particularly", "especially",
    "originally", "previously", "formerly", "currently", "recently",
    "constantly", "continuously", "repeatedly", "occasionally",
    "temporarily", "permanently", "normally", "typically", "usually",
    # --- 中文网文翻译高频副词 (CN Web Novel Translation Adverbs) ---
    "faintly", "vaguely", "seemingly", "surprisingly", "evidently",
    "helplessly", "speechlessly", "wordlessly", "unconsciously",
    "instinctively", "subconsciously", "reluctantly", "hesitantly",
    "cautiously", "curiously", "excitedly", "anxiously", "nervously",
    "furiously", "angrily", "coldly", "calmly", "indifferently",
    "casually", "lazily", "arrogantly", "disdainfully", "mockingly",
    "solemnly", "proudly", "stubbornly", "weakly", "bitterly",
    "incredulously", "doubtfully", "suspiciously", "impatiently",
    "eagerly", "gratefully", "regretfully", "sorrowfully", "joyfully",
    "cheerfully", "gloomily", "grimly", "sternly", "blankly",
    "abruptly", "briefly", "shortly", "afterward", "afterwards",
    "somewhat", "remarkably", "noticeably",
    "significantly", "considerably", "pleasantly", "unpleasantly",
    # --- 连接副词补充 (Connector Adverbs) ---
    "indeed", "possibly", "supposedly", "presumably",
    "additionally", "consequently", "subsequently", "accordingly",

    # --- 对话填充词 / 感叹词 (Dialogue Fillers & Interjections) ---
    "oh", "yes", "yeah", "okay", "right", "well", "like",
    "haha", "laugh", "bit",
    # --- 拟声词 (Onomatopoeia) ---
    "swoosh",
    "hmm", "uh", "ah", "hey", "ha", "um", "alright", "huh", "wow",
    "please", "thank", "thanks", "sorry", "fine", "sure", "hello",
    "damn", "god", "hell", "goodness", "heavens", "gosh",
    "dear", "ahem", "ooh", "hmph", "tsk", "sigh", "mhm", "nah",

    # --- 高频叙事名词 (High-frequency Narrative Nouns) ---
    "thing", "things", "people", "man", "woman", "way",
    "eyes", "hand", "hands", "head", "face", "door", "room", "voice",
    "person", "matter", "situation", "place", "inside", "moment",
    "sounds", "bag", "words", "palm",
    # --- 身体部位 (Body Parts — 小说中极高频) ---
    "body", "chest", "shoulder", "shoulders", "arm", "arms",
    "finger", "fingers", "lip", "lips", "mouth", "neck", "back",
    "feet", "foot", "heart", "throat", "eye", "ear", "ears",
    "skin", "hair", "knee", "knees", "leg", "legs", "forehead",
    "cheek", "cheeks", "chin", "jaw", "teeth", "tongue", "wrist",
    "elbow", "stomach", "belly", "waist", "nose", "bone", "bones",
    "fist", "fists", "thumb",
    # --- 空间/方向名词 (Spatial / Direction Nouns) ---
    "side", "ground", "floor", "wall", "air", "sky", "front",
    "edge", "corner", "center", "middle", "surface", "top", "bottom",
    "end", "direction", "distance", "spot", "area", "gap",
    "path", "road", "street", "entrance", "exit", "outside",
    "forward", "backward", "upward", "downward", "inward", "outward",
    "toward", "towards", "nearby", "aside", "beneath",
    "alongside", "throughout", "ahead", "overhead", "underneath",
    # --- 时间名词 (Time Nouns) ---
    "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "night", "nights", "morning", "evening", "afternoon",
    "tonight", "today", "yesterday", "tomorrow", "week", "weeks",
    "month", "months", "year", "years", "while", "instant",
    # --- 其他通用叙事名词 (Other Generic Narrative Nouns) ---
    "word", "sort", "kind", "fact", "rest", "lot", "deal",
    "couple", "half", "point", "idea", "reason", "sense",
    "sound", "form", "expression", "tone", "glow",
    "darkness", "breath", "steps",
    "sight", "smile", "nod",
    "silence", "noise", "sign", "line", "shape",
    "color", "colour", "piece", "group", "crowd", "pair",
    "attention", "effort",
    "trace", "hint", "aura", "presence",
    "appearance", "manner", "movement",
    "hesitation", "surprise", "shock", "anger",
    "relief", "joy", "sorrow", "grief", "pain", "pleasure",
    "impression", "intention", "notion",
    "speed", "force", "strength",
    "course", "result", "effect", "response", "reaction",
    "chance", "opportunity", "ability", "level",
    "condition", "position", "angle",
    "scene", "image", "picture", "view", "glimpse",
    "warning", "signal", "gesture", "motion",
    "brow", "brows", "grip", "grasp",
    "others",

    # --- 数词 (Numbers) ---
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand", "million", "billion",
    "dozen", "dozens", "several", "numerous", "countless",
    "once", "twice", "third", "fourth", "fifth",
    "first", "second", "last", "next",

    # --- 通用形容词 (Generic Adjectives) ---
    "another", "old", "new", "long", "little", "big", "good", "great",
    "small", "large", "dark", "white", "black", "red", "deep", "high",
    "low", "hard", "soft", "strong", "fast", "slow", "faint", "slight",
    "whole", "full", "single", "mere", "certain", "able", "unable",
    "different", "same", "possible", "impossible", "true", "wrong",
    "enough", "young", "bright", "clear", "warm", "cold", "hot",
    "thick", "thin", "wide", "narrow", "flat", "round", "sharp",
    "heavy", "strange", "familiar", "normal", "usual", "common",
    "important", "necessary", "obvious", "serious", "terrible", "difficult", "seamless",
    "horrible", "awful", "wonderful", "beautiful", "perfect",
    "ready", "afraid", "aware", "alone", "silent", "quiet",
    "calm", "gentle", "fierce", "wild", "desperate", "nervous",
    "curious", "surprised", "confused", "angry", "upset", "worried",
    "tired", "sick", "pale", "proper", "previous", "brief",
    # --- 情感/心理状态形容词 (Emotional State Adjectives) ---
    "stunned", "dumbfounded", "bewildered", "puzzled", "perplexed",
    "baffled", "astonished", "astounded", "amazed", "dazed",
    "horrified", "terrified", "petrified", "frightened", "scared",
    "furious", "enraged", "infuriated", "annoyed", "irritated",
    "frustrated", "disappointed", "embarrassed", "awkward",
    "uncomfortable", "helpless", "speechless", "relieved",
    "satisfied", "pleased", "delighted", "overjoyed",
    "determined", "focused", "exhausted", "shocked",
    # --- 通用描述形容词补充 (Generic Descriptive Adjectives) ---
    "vast", "immense", "enormous", "massive", "tiny", "huge",
    "endless", "utter", "sheer", "pure",
    "sudden", "rapid", "gradual", "steady", "constant",
    "intense", "powerful", "tremendous", "incredible", "unbelievable",
    "terrifying", "frightening", "astonishing", "astounding",
    "remarkable", "extraordinary", "rare", "empty", "dense",
    "rough", "smooth", "dull", "blank", "bare", "raw",
    "apparent", "evident", "visible", "invisible", "distant",
    "immediate", "direct", "indirect", "exact", "vague",
    "loose", "tight", "stiff", "rigid", "flexible",

    # --- 敏感/侮辱性词汇 ---
    "eunuch", "eunuchs", "whore", "whores",
    "bastard", "bastards", "bitch", "bitches",
    "slut", "sluts", "tramp", "tramps",
    "wench", "wenches", "hag", "hags", "crone", "crones",

    # --- 出版/电子书元数据 ---
    "gutenberg", "project", "ebook", "license",
    "foreword", "afterword", "introduction",

    # --- 中文小说拼音人名（高频噪声）---
    "chen", "zhangs", "xiang", "lei", "yang", "zhao", "xinhai", "qian",
    "ying", "lao", "yuanchao", "qiu", "hong", "jian", "hongji", "wukong",
    "xiyue", "xuhu", "qingyang", "zhu", "qi", "liu", "cheng", "kong",

    # --- 常见英文人名（跨书籍高频噪音）---
    # 男性名
    "john", "james", "jack", "david", "michael", "robert", "william", "richard",
    "thomas", "charles", "george", "edward", "henry", "peter", "daniel", "mark",
    "stephen", "andrew", "joseph", "adam", "ben", "tom", "bob", "bill", "jim",
    "joe", "mike", "nick", "luke", "alex", "matt", "tim", "chris", "fred",
    "carl", "arthur", "alan", "martin", "simon", "brian", "kevin", "patrick",
    "dennis", "gary", "larry", "jerry", "tony", "ralph", "walter",
    "harold", "albert", "ernest", "leonard", "victor", "bruce", "howard",
    "ian", "philip", "max", "leo", "roger", "eugene",
    # 女性名
    "alice", "mary", "jane", "elizabeth", "sarah", "ann", "anna", "anne",
    "margaret", "catherine", "helen", "emma", "lucy", "susan", "laura",
    "maria", "emily", "rachel", "ruth", "grace", "claire", "lily", "amy",
    "kate", "betty", "carol", "jean", "joan", "janet", "joyce", "judy",
    "karen", "linda", "lisa", "nancy", "barbara", "dorothy", "virginia",
    "martha", "gloria", "agnes", "ellen", "marie", "rebecca", "hannah",
    # 通用称谓
    "mr", "mrs", "ms", "sir", "lord", "lady", "dr", "professor",
    "king", "queen", "prince", "princess", "father", "mother", "brother",
    "sister", "uncle", "aunt", "son", "daughter",

    # --- 元数据相关词汇 ---
    "herbert",  # Dune相关元数据（dune已移除——书名/核心概念）
    "paperback", "bestseller", "trilogy",  # 出版信息
    "novels", "novel", "author", "writer", "writers", "wrote",  # 作者/作品相关
    "fiction", "literary", "poet", "poem",  # 文学相关
    "readers", "reader",  # 读者相关
    "copyright", "publisher", "published", "edition", "isbn",  # 出版信息
    "acknowledgments", "acknowledgements", "dedication",  # 书籍元数据
    "www", "http", "https", "com", "org",  # URL残留

    # ================= 4. 疑问词 (Question Words) =================
    "who", "what", "why", "where", "when", "which", "whose", "whom",
    "how", "whether", "if",

    # ================= 5. 常见英文停用词 (Common English Stop Words) =================

    # --- 代词 (Pronouns) ---
    # 人称代词 (Personal Pronouns)
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    # 指示代词 (Demonstrative Pronouns)
    "this", "that", "these", "those",
    # 不定代词 (Indefinite Pronouns)
    "one", "oneself", "ones",
    "someone", "somebody", "something",
    "anyone", "anybody", "anything",
    "everyone", "everybody", "everything",
    "nobody", "no one", "nothing",
    "another", "others", "other",
    "each", "either", "neither",
    "any", "all", "both", "few", "many", "most", "some", "none",
    "several", "enough", "plenty",
    # 关系代词 / 疑问代词 (Relative / Interrogative)
    "who", "whom", "whose", "which", "what", "that",
    "whoever", "whomever", "whatever", "whichever",

    # --- 限定词和量词 (Determiners & Quantifiers) ---
    "the", "a", "an",
    "every", "each", "no", "any", "some", "all", "both",
    "such", "certain", "various", "whole", "entire",
    "numerous", "much", "more", "most", "less", "least", "fewer",
    "little", "own", "same", "other", "another",

    # --- 介词 (Prepositions) ---
    "with", "in", "on", "at", "by", "for", "from", "to", "of", "about",
    "into", "onto", "upon", "over", "under", "above", "below",
    "between", "among", "amongst", "through", "during", "before", "after",
    "since", "until", "till", "within", "without",
    "against", "across", "around", "behind", "beside", "besides",
    "beyond", "near", "off", "out", "up", "down",
    "along", "toward", "towards", "inside", "outside",
    "underneath", "beneath", "throughout", "alongside",
    "amid", "amidst", "atop", "via", "per",
    "like", "unlike", "worth", "except",
    "concerning", "regarding", "considering", "following",
    "given", "including", "excluding", "plus", "minus", "versus",

    # --- 连词 (Conjunctions) ---
    "and", "or", "but", "so", "yet", "nor",
    "as", "than", "that", "if", "once",
    "while", "although", "though", "even though",
    "because", "since", "unless", "until", "when", "whenever",
    "where", "wherever", "whereas", "whether",
    "provided", "lest", "except that",
    "due",  # 语法碎片（due to）

    # --- 助动词和情态动词 (Auxiliary and Modal Verbs) ---
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should",
    "could", "can", "may", "might", "must",
    "ought", "need", "dare",

    # --- 缩写残留 (Contraction Fragments) ---
    "didn", "wasn", "couldn", "isn", "aren", "doesn", "hasn", "haven",
    "wouldn", "shouldn", "don", "won", "hadn", "weren", "ain",
    "ve", "re", "ll", "nt", "mustn", "needn", "shan",
    "didn't", "wasn't", "couldn't", "isn't", "aren't", "doesn't",
    "hasn't", "haven't", "wouldn't", "shouldn't", "don't", "won't",
    "hadn't", "weren't", "can't", "cannot", "mustn't", "needn't",
    "shan't", "it's", "i'm", "he's", "she's", "we're", "they're",
    "i've", "we've", "they've", "you've", "i'll", "we'll", "they'll",
    "you'll", "he'll", "she'll", "it'll", "i'd", "we'd", "they'd",
    "you'd", "he'd", "she'd", "that's", "what's", "who's", "there's",
    "here's", "let's", "ain't",

    # --- 常见副词 (Common Adverbs) ---
    # 频率副词 (Frequency)
    "always", "never", "ever", "often", "sometimes", "usually",
    "rarely", "seldom", "frequently", "occasionally", "constantly",
    # 时间副词 (Time)
    "now", "then", "already", "still", "yet", "soon", "ago",
    "recently", "lately", "immediately", "suddenly", "finally",
    "eventually", "previously", "formerly", "afterwards", "meanwhile",
    "once", "twice",
    # 程度副词 (Degree)
    "very", "too", "quite", "rather", "really", "fairly", "pretty",
    "almost", "nearly", "barely", "hardly", "scarcely",
    "completely", "entirely", "totally", "absolutely", "utterly",
    "simply", "merely", "just", "even", "enough",
    # 方式/态度副词 (Manner / Attitude)
    "also", "again", "perhaps", "maybe", "probably", "possibly",
    "certainly", "definitely", "surely", "indeed", "obviously",
    "clearly", "apparently", "naturally", "actually", "basically",
    "essentially", "generally", "particularly", "especially", "specifically",
    "exactly", "partly", "mainly", "mostly", "largely", "primarily",
    # 方位/方向副词 (Direction / Place)
    "here", "there", "away", "back", "far", "forth", "forward",
    "ahead", "apart", "aside", "together",
    "anywhere", "everywhere", "somewhere", "nowhere",
    # 其他副词 (Other)
    "else", "anyway", "anyhow", "somehow", "somewhat", "anymore",
    "otherwise", "instead", "nevertheless", "nonetheless",
    "furthermore", "moreover", "therefore", "hence", "thus",
    "however", "consequently", "accordingly", "regardless",
    "likewise", "similarly", "conversely", "alternatively",
    "not", "no", "only",

    # --- 高频叙事动词 (High-Frequency Narrative Verbs) ---
    # 说/问 (Speech)
    "say", "said", "says", "saying",
    "tell", "told", "tells", "telling",
    "ask", "asked", "asks", "asking",
    "call", "called", "calls", "calling",
    "reply", "replied",
    "answer", "answered",
    "speak", "spoke", "spoken", "speaks", "speaking",
    "talk", "talked", "talks", "talking",
    "mention", "mentioned",
    "whisper", "whispered",
    "shout", "shouted",
    "cry", "cried",
    "mutter", "muttered",
    "murmur", "murmured",
    "exclaim", "exclaimed",
    # 看/知/想 (Perception & Cognition)
    "see", "saw", "seen", "sees", "seeing",
    "look", "looked", "looks", "looking",
    "watch", "watched",
    "stare", "stared",
    "glance", "glanced",
    "gaze", "gazed",
    "notice", "noticed",
    "know", "knew", "known", "knows", "knowing",
    "think", "thinks",
    "feel", "felt", "feels", "feeling",
    "believe", "believed",
    "wonder", "wondered",
    "realize", "realized",
    "understand", "understood",
    "suppose", "supposed",
    "guess", "guessed",
    "seem", "seemed", "seems", "seeming",
    "appear", "appeared", "appears",
    "hear", "heard", "hears", "hearing",
    # 移动 (Movement)
    "go", "went", "gone", "goes", "going",
    "come", "came", "comes", "coming",
    "move", "moved", "moves", "moving",
    "walk", "walked", "walks", "walking",
    "run", "ran", "runs", "running",
    "turn", "turned", "turns", "turning",
    "step", "stepped",
    "follow", "followed",
    "leave", "left", "leaves", "leaving",
    "return", "returned",
    "enter", "entered",
    "reach", "reached",
    "pass", "passed",
    "cross", "crossed",
    # 取/给/持 (Manipulation)
    "get", "got", "gotten", "gets", "getting",
    "take", "took", "taken", "takes", "taking",
    "give", "gave", "given", "gives", "giving",
    "make", "made", "makes", "making",
    "put", "puts", "putting",
    "set", "sets", "setting",
    "hold", "held", "holds", "holding",
    "keep", "kept", "keeps", "keeping",
    "let", "lets", "letting",
    "bring", "brought", "brings", "bringing",
    "pull", "pulled",
    "push", "pushed",
    "pick", "picked",
    "drop", "dropped",
    "throw", "threw", "thrown",
    "catch", "caught",
    "carry", "carried",
    "send", "sent",
    "draw", "drew", "drawn",
    # 姿态/身体 (Posture & Body)
    "stand", "stood", "stands", "standing",
    "sit", "sat", "sits", "sitting",
    "lie", "lay", "lain", "lying",
    "rise", "rose", "risen",
    "fall", "fell", "fallen",
    "nod", "nodded",
    "shake", "shook", "shaken",
    "smile", "smiled",
    "laugh", "laughed",
    "sigh", "sighed",
    # 一般动作 (General Actions)
    "do", "did", "does", "doing", "done",
    "find", "found", "finds", "finding",
    "want", "wanted", "wants", "wanting",
    "need", "needed", "needs", "needing",
    "try", "tried", "tries", "trying",
    "use", "used", "uses", "using",
    "start", "started", "starts", "starting",
    "begin", "began", "begun", "begins", "beginning",
    "stop", "stopped",
    "continue", "continued",
    "open", "opened",
    "close", "closed",
    "show", "showed", "shown", "shows", "showing",
    "wait", "waited",
    "happen", "happened", "happens",
    "become", "became", "becomes", "becoming",
    "mean", "meant", "means", "meaning",
    "remember", "remembered",
    "manage", "managed",
    "remain", "remained",
    "seem", "seemed",

    # --- 疑问词 (Question Words) ---
    "why", "how", "whether",

    # --- 其他高频功能性词汇 (Other Common Function Words) ---
    "yes", "no", "not", "ok", "okay", "please", "thank", "thanks",
    "well", "oh", "ah", "um", "uh", "hm", "hmm",
    "like", "just", "even", "still", "already", "also",
    "despite", "although", "though",
    "ago", "per", "via",
    "able", "unable",
    "way", "thing", "things", "lot", "bit",

    # --- 补充人名 (Additional Character Names) ---
    "xinyue", "er", "lan", "sylvia", "xiaohou",
    "kylie", "emlyn", "belinda", "olivier", "ojwin",
    "hai",

    # --- 无差别空间词 (Generic Spatial / Location Words) ---
    "city", "world", "land", "plains", "house", "street", "bank",
    "room", "door", "wall", "ground", "sky", "surroundings",


    # ================= 6. Topic Modeling 可视化中发现的人名噪音 =================
    # 以下为确认的「人名/角色名」，在 BERTopic 的 topic keywords 中出现后加入。
    # 添加前须逐项核实，禁止加入：
    #   - 普通英语词（如 tyrian=提尔的、bridge=桥）
    #   - 学科术语（如 orogeny=造山运动、axon=轴突）
    #   - 中文拼音/义项（如 dan=丹、ling=灵）
    #   - 仅概念/地名且非人名的虚构词（如 godshatter 等）
    # 按来源分组，避免重复添加已有条目。

    # ------ 追加 ASOIAF / Game of Thrones 角色 ------
    "redwyne", "oberyn", "clegane", "slynt", "grenn", "donal",
    "jhogo", "kraznys", "jhiqui", "gendry", "amory", "meryn",
    "chiswyck", "shireen", "renly", "salladhor", "drogon",
    "walders", "cerwyn", "stormcrows", "mummers", "maester",

    # ------ 追加 Harry Potter 角色 ------
    "neville", "seamus", "viktor", "dursleys", "trelawney",
    "ravenclaw", "muggle",

    # ------ 追加 Foundation / Asimov 角色 ------
    "daneel", "pel", "fallom", "acarnio", "joranumite",
    "trantorian", "dahlite", "sayshell", "sayshellian", "comporellon",

    # ------ 追加 Pratchett (Making Money 等) 角色 ------
    "vimes", "dearheart", "gladys", "cribbins", "sacharissa",
    "spangler", "flead",

    # ------ 追加 Vorkosigan Saga 角色 ------
    "kanzian", "cordelias", "droushnakovi", "kareen", "padma",
    "naismith", "overholt", "framingham", "jacksonian", "elena",

    # ------ 追加 Tolkien (Silmarillion 等) 角色 ------
    "morwen", "felagund", "aragorn", "gelion", "brethil",
    "nargothrond", "tirion",

    # ------ 追加 Broken Earth / Jemisin 角色（仅人名/角色名，不含 orogeny 等地质术语）------
    # orogene/orogenes 已移除——《破碎的地球》核心概念词
    "lerna", "syen", "castrimans", "castrima",
    "rennies",

    # ------ 追加 Mars Trilogy / Kim Stanley Robinson 角色 ------
    "bithras", "zeyk", "jackie", "spencer",

    # ------ 追加 Three-Body Problem 特有词 ------
    # sophon/sophons/wallfacers 已移除——《三体》核心概念词

    # ------ 追加 Connie Willis 时间旅行系列角色 ------
    "mesiel", "chattisbourne", "runnymede", "skendgate",
    "ahrens", "badri",

    # ------ 追加 Vernor Vinge 角色 ------
    "flenserists", "vendacious", "samnorsk", "blueshell",

    # ------ 追加 China Miéville 角色 ------
    "ragamoll", "susullil", "pennyhaugh",

    # ------ 追加 Ancillary Justice / Ann Leckie 角色 ------
    "garseddai", "garsedd", "sarrse", "skaaiat", "vendaai",
    "alanye", "katish", "inye", "effegen", "konye", "pippashap",

    # ------ 追加 Lord Valentine's Castle / Silverberg 角色 ------
    "narrameer", "piurifayne", "brangalyn", "malibor",
    "ilirivoyne", "lisamon", "velalisier",

    # ------ 追加 Uplift War / David Brin 角色 ------
    "gailet", "uthacalthing", "thennanin", "tymbrimi", "garthlings",

    # ------ 追加 Yiddish Policemen's Union / Chabon 角色 ------
    "tenenboym", "rudashevsky", "yakovy", "gelbfish", "verbovers",

    # ------ 追加 Dreamsnake / McIntyre 角色 ------
    "merideth", "lainie", "pauli",

    # ------ 追加 Harpist in the Wind / McKillip 角色 ------
    "akren", "astrin", "caerweddin", "talies", "yrth",

    # ------ 追加 Diamond Age / Neal Stephenson 角色 ------
    "phyles", "hackworths", "gwendolyn", "gwen", "cocklebur",
    "chevaline",

    # ------ 追加 Stand on Zanzibar / John Brunner 角色 ------
    "yatakang", "yatakangi", "solukarta", "hamilcar",

    # ------ 追加 Altered Carbon / Richard K. Morgan 角色 ------
    "hendrix", "davidson", "elias", "rutherford", "prescott",

    # ------ 追加 Bitter Angels / C.L. Anderson 角色 ------
    "erasmans", "amerand", "torian", "felice", "jerimiah", "solarans",

    # ------ 追加 Emissaries from the Dead 角色 ------
    "lastogne", "porrinyard", "porrinyards", "hammocktown", "skye",

    # ------ 追加 Ringworld / Larry Niven 角色 ------
    "halrloprillalar", "prill", "hindmost",

    # ------ 追加 Quantum Rose / Catherine Asaro 角色 ------
    "ironbridge", "azander", "tulain", "morlin", "sunsmith",

    # ------ 追加 These Burning Stars / Bethany Jacobs 角色 ------
    "knowlessyndicate",

    # ------ 追加 Downbelow Station / C.J. Cherryh 角色 ------
    "konstantins", "mazian", "miliko",

    # ------ 追加 Ender's Game 角色 ------
    "rackham", "locke",

    # ------ 追加其他书籍特有角色 ------
    "tarosse", "blumer",  # Monster Paradise
    "valenda",  # Goblin Emperor
    "herne", "miroe", "mantagnes",  # various
    "kapoor", "galton", "lev",  # detective stories
    "galbreath", "shawbeck",  # 253
    "curriden", "pharram", "jaymac", "hellbenders",  # Alvin Maker
    "sloan",  # various
    "campion", "petrus", "lindley", "godwin",  # various（tyrian=英语词“提尔的”已排除）
    "bowden", "khurusch", "aikam", "yorjavic", "buidze",  # various
    "martindale", "hernan",  # various
    "bartok", "kinte", "kingsley", "mercer",  # various
    "brankov", "winton", "kirilen", "arne", "wescotts", "shara",  # Gateway/various
    "mondragon", "zasha", "ygassdril",  # 2312
    "vaughan", "reuben", "bolbay", "lurt", "klast",  # Hominids
    "rosemund", "roche",  # Book of New Sun
    "brone",  # Ringworld/ASOIAF
    "marek",  # Uprooted
    "abernathy",  # Redshirts
    "anatoly",  # Red Mars
    "chowdhury", "satie", "lautagata", "stevens",  # various
    "wyles", "lott",  # 253/various
    "salvoy",  # Last Call
    "jermyn", "theophilus", "winnifred",  # Anubis Gates
    "rheinhardt",  # various
    "hoddling", "curhouse",  # Saint of Bright Doors
    "hejmen",  # Saint of Bright Doors
    "wargin",  # various
    "houdina",  # Middlegame
    "crowley",  # American Gods / Good Omens
    "maecar", "doral",  # Gods Themselves
    "shimon",  # 253
    "wulfyre", "wulfyres",  # Someone You Can Build a Nest In
    "shadry", "arshadin", "rosseth",  # Innkeeper's Song
    "bossies", "rogan", "hoxworth",  # various
    "noyes", "reiss", "heydrich", "hassop",  # various
    "svensndot", "bergsndot", "grondr",  # Fire Upon Deep（godshatter=概念词已排除）
    "hanse",  # Fire Upon Deep
    "rangan",  # Apex（axon=神经轴突术语已排除）
    "mondragon",  # 2312
    "woolcombe",  # various
    "stanny", "vizzy",  # Software
    "maedda", "northsetting", "sadik", "tirin", "rulag",  # Dispossessed
    "shiren",  # Witch King
    "hanari",  # various
    "watkiss",  # various

    # ------ 追加常见英文人名（topic keywords 中高频出现但未覆盖）------
    # 注意：dan 不加入——在网文语料中多为“丹”(elixir)，非英文男名
    # 男性名
    "colin", "jonas", "jake", "grayson", "andy", "duncan",
    "damian", "harley", "oliver", "leigh", "darren", "clive", "hal",
    "jimmy", "boris", "tristan", "tyler", "gideon", "liam", "winston",
    "alfred", "matthew", "napoleon", "stuart", "shaw", "taylor",
    "hendrick", "currie", "reuben", "locke", "finch", "hercules",
    "marek", "godwin", "ramsey", "casimir",
    # 女性名
    "jenny", "sandra", "ursula", "phyllis", "jackie", "anya",
    "dorothea", "carrie", "carmen", "lorraine", "marge", "judith",
    "felicity", "stella", "abby", "maud", "eliza", "katie",
    "frankie", "rita", "tess", "penelope", "miriam", "joni",
    "harriet", "sadie", "gemma", "dolores", "lindy", "lorelei",
    "geraldine", "winnie", "lena", "cora", "mona", "fiona",
    "ariana", "vera", "melanie", "jasper", "bella", "brynn",
    "susanna", "gwendolyn", "gwen", "mae", "yolanda", "megan",
    "erin", "kim", "stevie", "bronwen", "guinevere", "lyra",
    "sunita", "eden", "valentina",
]