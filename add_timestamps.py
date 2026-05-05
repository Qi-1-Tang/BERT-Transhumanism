"""
为 bert_training_dataset.csv 添加 Timestamps 列
根据5个MD文件中的出版/完结时间，为每本书的每个文本片段添加时间戳

用法:
  python add_timestamps.py --input /root/autodl-tmp/clean/bert_training_dataset.csv --output bert_training_dataset.csv
"""

import os
import re
import argparse
import pandas as pd


# ============================================================
# 220本书 → 年份 的映射表
# 数据来源: 5个MD文件（中文小说完结时间、Hugo奖、Nebula奖、Locus奖、菲利普迪克奖）
# ============================================================

BOOK_TIMESTAMPS = {
    # ==================== 中文网络小说（20本）====================
    # 完结/主要连载年份
    "Desolate Era": 2016,
    "I Shall Seal the Heavens": 2015,
    "Library of Heaven": 2019,       # Library of Heaven's Path
    "Lord of the Mysteries": 2020,
    "Stellar Transformations": 2008,
    "Legendary Mechanic": 2020,      # The Legendary Mechanic / 超神机械师
    "Reverend Insanity": 2019,       # 蛊真人（2019年被封）
    "Tales of Demons and Gods": 2018, # 妖神记（未完结，以活跃期计）
    "Cultivation Chat Group": 2020,
    "True Martial World": 2018,      # 真武世界（武极天下续作，2018完结）
    "Battle Through the Heavens": 2012, # 斗破苍穹
    "Soul Land": 2011,               # 斗罗大陆
    "Warlock of the Magus World": 2017,
    "Monster Paradise": 2019,
    "Versatile Mage": 2020,          # 全职法师
    "Swallowed Star": 2012,
    "Forty Millenniums of Cultivation": 2019,
    "A Will Eternal": 2017,
    "Coiling Dragon": 2009,
    "Pocket Hunting Dimension": 2019,
    "Renegade Immortal": 2013,

    # ==================== Hugo奖获奖小说（75本）====================
    "A Canticle for Leibowitz": 1961,
    "A Deepness in the Sky": 2000,
    "A Desolation Called Peace": 2022,
    "A Fire Upon the Deep": 1993,
    "A Memory Called Empire": 2020,
    "All Clear": 2011,
    "American Gods": 2002,
    "Among Others": 2012,
    "Ancillary Justice": 2014,
    "Barrayar": 1992,
    "Blackout": 2011,
    "Blue Mars": 1997,
    "Case of Conscience": 1959,       # A Case of Conscience
    "City and the City": 2010,        # The City and the City
    "Cyteen": 1989,
    "Diamond Age": 1996,              # The Diamond Age
    "Doomsday Book": 1993,
    "Double Star": 1956,
    "Downbelow Station": 1982,
    "Dreamsnake": 1979,
    "Dune": 1966,
    "Ender": 1986,                    # Ender's Game
    "Fahrenheit 451": 1954,
    "Forever Peace": 1998,
    "Forever War": 1976,              # The Forever War
    "Gateway": 1978,
    "Green Mars": 1994,
    "Harry Potter": 2001,             # 用于 Goblet of Fire 的匹配
    "Hominids": 2003,
    "Hyperion": 1990,
    "Jonathan Strange": 2005,         # Jonathan Strange & Mr. Norrell
    "Lord of Light": 1968,
    "Mirror Dance": 1995,             # Mirror dance
    "Nettle": 2023,                   # Nettle & Bone
    "Network Effect": 2021,
    "Neuromancer": 1985,
    "Obelisk Gate": 2017,             # The Obelisk Gate
    "Paladin of Souls": 2004,
    "Rainbow": 2007,                  # Rainbow's End
    "Redshirts": 2013,
    "Rendezvous with Rama": 1974,
    "Ringworld": 1971,
    "Some Desperate Glory": 2024,
    "Speaker for the Dead": 1987,
    "Spin": 2006,                     # Spin - Robert Charles Wilson
    "Stand on Zanzibar": 1969,
    "Starship Troopers": 1960,
    "Startide Rising": 1984,
    "Stranger in a Strange Land": 1962,
    "Big Time": 1958,                 # The Big Time
    "Calculating Stars": 2019,        # The Calculating Stars
    "Demolished Man": 1953,           # The Demolished Man
    "Dispossessed": 1975,             # The Dispossessed
    "Fifth Season": 2016,             # The Fifth Season
    "Foundation": 1953,               # The Foundation Trilogy
    "Fountains of Paradise": 1980,    # The Fountains of Paradise
    "Gods Themselves": 1973,          # The Gods Themselves
    "Graveyard Book": 2009,           # The Graveyard Book
    "Left Hand of Darkness": 1970,    # The Left Hand of Darkness
    "Man in the High Castle": 1963,   # The Man in the High Castle
    "Moon Is a Harsh Mistress": 1967, # The Moon Is a Harsh Mistress
    "Snow Queen": 1981,               # The Snow Queen
    "Stone Sky": 2018,                # The Stone Sky
    "Tainted Cup": 2025,              # The Tainted Cup
    "Three-Body Problem": 2015,       # The Three-Body Problem (Hugo获奖年份)
    "Uplift War": 1988,               # The Uplift War
    "Vor Game": 1991,                 # The Vor Game
    "Wanderer": 1965,                 # The Wanderer (Fritz Leiber)
    "Yiddish Policemen": 2008,        # The Yiddish Policemen's Union
    "Rather Be Right": 1955,          # They'd Rather Be Right
    "This Immortal": 1966,
    "To Say Nothing of the Dog": 1999,
    "Scattered Bodies": 1972,         # To Your Scattered Bodies Go
    "Way Station": 1964,
    "Where Late the Sweet Birds Sang": 1977,

    # ==================== Nebula奖获奖小说（34本）====================
    "2312": 2013,
    "Master of Djinn": 2022,          # A Master of Djinn
    "Song for a New Day": 2020,       # A Song for a New Day
    "Time of Changes": 1972,          # A Time of Changes
    "Annihilation": 2015,
    "Babel": 2023,                    # Babel, or the Necessity of Violence
    "Babel-17": 1967,
    "Camouflage": 2005,
    "Darwin": 2000,                   # Darwin's Radio
    "Falling Free": 1989,             # Falling free (Vorkosigan Saga)
    "Flowers for Algernon": 1967,
    "Man Plus": 1977,
    "Moving Mars": 1995,
    "No Enemy But Time": 1983,
    "Parable of the Talents": 2000,
    "Powers": 2009,                   # Powers (Ursula K. Le Guin)
    "Red Mars": 1994,
    "Rite of Passage": 1969,
    "Seeker": 2007,
    "Slow River": 1997,
    "Someone You Can Build a Nest In": 2025,
    "Stations of the Tide": 1992,     # Stations Of The Tide
    "Tehanu": 1991,
    "Book of the New Sun": 1982,      # The Book of the New Sun
    "Einstein Intersection": 1968,    # The Einstein Intersection
    "Falling Woman": 1988,            # The Falling Woman
    "Healer": 1990,                   # The Healer's War
    "Moon and the Sun": 1998,         # The Moon and the Sun
    "Quantum Rose": 2002,             # The Quantum Rose
    "Saint of Bright Doors": 2024,    # The Saint of Bright Doors
    "Terminal Experiment": 1996,      # The Terminal Experiment / terminal experiment
    "Timescape": 1981,
    "Uprooted": 2016,
    "All the Birds in the Sky": 2017,

    # ==================== Locus奖获奖小说（47本，去重后新增）====================
    "Alvin Journeyman": 1996,         # Alvin journeyman
    "Anansi Boys": 2006,
    "Clash of Kings": 1999,           # A Clash of Kings
    "Dance with Dragons": 2012,       # A Dance with Dragons
    "Sorceress Comes to Call": 2024,  # A Sorceress Comes to Call
    "Storm of Swords": 2001,          # A Storm of Swords
    "Beauty": 1992,                   # Beauty (Sheri S. Tepper)
    "Brittle Innings": 1995,
    "Earthquake Weather": 1998,
    "Game of Thrones": 1997,          # A Game of Thrones
    "Harpist in the Wind": 1980,      # Harpist in the wind
    "Harry Potter": 2000,             # 此条被Harry Potter覆盖（Prisoner of Azkaban: 2000 / Goblet of Fire: 2001，取2001）
    "Iron Council": 2005,
    "Jade Legacy": 2022,
    "Job": 1985,                      # Job: A Comedy of Justice
    "Last Call": 1993,
    "Lavinia": 2009,
    "Lord Valentine": 1981,           # Lord Valentine's Castle
    "Making Money": 2008,
    "Middlegame": 2020,
    "Prentice Alvin": 1990,
    "Red Prophet": 1989,
    "Seventh Son": 1988,
    "Silmarillion": 1978,             # The Silmarillion
    "Soldier of the Mist": 1987,
    "Spinning Silver": 2019,
    "Scar": 2003,                     # The Scar
    "City We Became": 2021,           # The City We Became
    "Claw of the Conciliator": 1982,  # The Claw Of The Conciliator
    "Goblin Emperor": 2015,           # The Goblin Emperor
    "Innkeeper": 1994,                # The Innkeeper's Song
    "Mists of Avalon": 1984,          # The Mists of Avalon
    "Ocean at the End of the Lane": 2014, # The Ocean at the End of the Lane
    "Privilege of the Sword": 2007,   # The Privilege of the Sword
    "Trumps of Doom": 1986,
    "Witch King": 2024,
    "Kaiju Preservation Society": 2023, # The Kaiju Preservation Society
    "System Collapse": 2024,
    "City in the Middle of the Night": 2020, # The City in the Middle of the Night
    "Man Who Saw Seconds": 2016,      # The Man Who Saw Seconds

    # ==================== 菲利普·迪克奖获奖小说（44本，去重后新增）====================
    "253": 1998,                      # 253: The Print Remix
    "Altered Carbon": 2003,
    "Apex": 2014,
    "Bannerless": 2018,
    "Bitter Angels": 2011,
    "Countdown City": 2014,
    "Dead Space": 2022,
    "Dinner at Deviant": 1986,        # Dinner at Deviant's Palace
    "Elvissey": 1994,
    "Emissaries from the Dead": 2009,
    "Four Hundred Billion Stars": 1989,
    "Growing Up Weightless": 1994,
    "Headcrash": 1996,
    "Homunculus": 1986,
    "King of Morning": 1992,          # King of Morning, Queen of Day
    "Life": 2005,                     # Life (Gwyneth Jones)
    "Lost Everything": 2013,
    "Mysterium": 1995,
    "Nova Swing": 2007,
    "Only Forward": 1995,
    "Points of Departure": 1991,
    "Road Out of Winter": 2021,
    "Ship of Fools": 2002,
    "Software": 1983,
    "Sooner or Later Everything Falls": 2020, # Sooner or Later Everything Falls into the Sea
    "Spin Control": 2007,
    "Strange Toys": 1988,
    "Subterranean Gallery": 1990,
    "Terminal Mind": 2013,
    "Anubis Gates": 1984,             # The Anubis Gates
    "Book of the Unnamed Midwife": 2016, # The Book of the Unnamed Midwife
    "Extractionist": 2021,            # The Extractionist
    "Mercy Journals": 2017,           # The Mercy Journals
    "Mount": 2003,                    # The Mount
    "Strange Affair of Spring Heeled Jack": 2011, # The Strange Affair of Spring Heeled Jack
    "Time Ships": 1996,               # The Time Ships
    "Troika": 1998,                   # The Troika
    "Theory of Bastards": 2019,
    "These Burning Stars": 2024,
    "Through the Heart": 1993,
    "Time's Agent": 2024,             # Time's Agent
    "Times Agent": 2024,              # Time's Agent (无撇号变体)
    "Vacuum Diagrams": 1998,
    "War Surf": 2006,
    "Wetware": 1989,
}

# Harry Potter 特殊处理: Goblet of Fire (Hugo 2001), Prisoner of Azkaban (Locus 2000)
# CSV中有两个: Harry_Potter_3... (Prisoner) 和 Harry Potter the Goblet of Fire
# 后面用特殊逻辑区分


def normalize(s):
    """将字符串标准化用于匹配：小写、去标点、多空格合一"""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def match_book_to_year(book_name):
    """
    根据 book_name（CSV中的文件名）匹配到出版/获奖年份
    返回 (year, matched_key) 或 (None, None)
    """
    norm = normalize(book_name)
    
    # ---- 特殊处理 ----
    # Harry Potter: Prisoner of Azkaban vs Goblet of Fire
    if "harry potter" in norm:
        if "prisoner" in norm or "harry_potter_3" in norm.replace(' ', '_'):
            return 2000, "Harry Potter (Prisoner of Azkaban)"
        elif "goblet" in norm:
            return 2001, "Harry Potter (Goblet of Fire)"
        else:
            return 2001, "Harry Potter"
    
    # Spin vs Spin Control: 先检查更长的匹配
    if "spin control" in norm:
        return 2007, "Spin Control"
    
    # "All the Birds in the Sky" - 可能有多种格式
    if "all the birds in the sky" in norm:
        return 2017, "All the Birds in the Sky"
    
    # 对于 "Spin" 要避免误匹配 "Spinning Silver"
    if "spinning silver" in norm:
        return 2019, "Spinning Silver"
    
    # Complete Book of the New Sun (合集) - 含有 Claw of the Conciliator
    if "complete book of the new sun" in norm:
        return 1982, "Book of the New Sun"
    
    # Foundation 合集
    if "foundation" in norm:
        return 1953, "Foundation"
    
    # 尝试从长到短匹配所有key
    # 按key长度降序排列，优先匹配更具体的key
    sorted_keys = sorted(BOOK_TIMESTAMPS.keys(), key=lambda k: len(k), reverse=True)
    
    for key in sorted_keys:
        norm_key = normalize(key)
        if norm_key in norm:
            return BOOK_TIMESTAMPS[key], key
    
    return None, None


def add_timestamps_to_csv(input_path, output_path):
    """为CSV添加Timestamps列"""
    print(f"读取CSV: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    print(f"总行数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    
    if 'book_name' not in df.columns:
        raise ValueError("CSV必须包含 'book_name' 列")
    
    # 获取所有唯一的book_name
    unique_books = df['book_name'].unique()
    print(f"唯一书名数量: {len(unique_books)}")
    
    # 建立 book_name → year 的映射
    book_year_map = {}
    matched = 0
    unmatched = []
    
    for book_name in unique_books:
        year, key = match_book_to_year(str(book_name))
        if year is not None:
            book_year_map[book_name] = year
            matched += 1
        else:
            unmatched.append(book_name)
    
    print(f"\n匹配结果: {matched}/{len(unique_books)} 本书匹配成功")
    
    if unmatched:
        print(f"\n 以下 {len(unmatched)} 本书未能匹配到时间戳:")
        for bn in unmatched:
            print(f"  - {bn[:80]}...")
    
    # 添加 Timestamps 列
    df['Timestamps'] = df['book_name'].map(book_year_map)
    
    # 统计
    na_count = df['Timestamps'].isna().sum()
    valid_count = len(df) - na_count
    print(f"\n时间戳覆盖: {valid_count}/{len(df)} 行有时间戳 ({valid_count/len(df)*100:.1f}%)")
    
    if na_count > 0:
        print(f"  {na_count} 行缺少时间戳（来自未匹配的书）")
    
    # 显示时间分布
    print("\n时间分布:")
    year_dist = df['Timestamps'].dropna().value_counts().sort_index()
    for year, count in year_dist.items():
        print(f"  {int(year)}: {count} 个片段")
    
    # 保存
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n已保存到: {output_path}")
    print(f"新增列: Timestamps（出版/完结年份）")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='为 bert_training_dataset.csv 添加时间戳列')
    parser.add_argument('--input', type=str, default='bert_training_dataset.csv',
                        help='输入CSV路径（默认: bert_training_dataset.csv）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV路径（默认: 覆盖输入文件）')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input
    
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 {args.input}")
        return
    
    add_timestamps_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
