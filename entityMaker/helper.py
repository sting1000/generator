from generator.normalizer import Normalizer
import pandas as pd
import random


def get_item(value, item_type, lan, aliases=None):
    if aliases is None:
        aliases = []
    item = {
        "value": value,
        "type": item_type,
        "language": lan,
        "normalizedValue": Normalizer().normalize_text(value, lan),
        "aliases": aliases
    }
    return item


def make_seconds(entity_type, lan, entity_amount=None, random_seed=42):
    all_list = []
    #  random.seed(random_seed)

    if lan == "en":
        for num in range(2, 60):
            all_list.append(get_item("{} seconds".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 second", entity_type, lan, aliases=["a minute"]))
        all_list.append(get_item("half a second", entity_type, lan, aliases=["half second"]))
    elif lan == "de":
        for num in range(2, 60):
            all_list.append(get_item("{} sekunden".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 sekunde", entity_type, lan))
        all_list.append(get_item("einen halbe sekunde", entity_type, lan))
    elif lan == "fr":
        for num in range(2, 60):
            all_list.append(get_item("{} secondes".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 seconde", entity_type, lan))
        all_list.append(get_item("une demi seconde", entity_type, lan))
    elif lan == "it":
        for num in range(2, 60):
            all_list.append(get_item("{} secondi".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 secondo", entity_type, lan))
        all_list.append(get_item("mezzo secondo", entity_type, lan))
    else:
        print("ERROR: Wrong language!")

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.sample(all_list, k=entity_amount)


def make_minutes(entity_type, lan, entity_amount=None, random_seed=42):
    """
    entity_amount: An integer defining the length of the returned list
    """
    all_list = []
    #  random.seed(random_seed)

    if lan == "en":
        for num in range(2, 60):
            all_list.append(get_item("{} minutes".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 minute", entity_type, lan, aliases=["a minute"]))
        all_list.append(get_item("half a minute", entity_type, lan, aliases=["half minute"]))
    elif lan == "de":
        for num in range(2, 60):
            all_list.append(get_item("{} minuten".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 minute", entity_type, lan))
        all_list.append(get_item("einen halbe minute", entity_type, lan))
    elif lan == "fr":
        for num in range(2, 60):
            all_list.append(get_item("{} minutes".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 minute", entity_type, lan))
        all_list.append(get_item("une demi minute", entity_type, lan))
    elif lan == "it":
        for num in range(2, 60):
            all_list.append(get_item("{} minuti".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 minuto", entity_type, lan))
        all_list.append(get_item("mezzo minuto", entity_type, lan))
    else:
        print("ERROR: Wrong language!")

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.sample(all_list, k=entity_amount)


def make_hours(entity_type, lan, entity_amount=None, random_seed=42):
    """
    entity_amount: An integer defining the length of the returned list
    """
    all_list = []
    # random.seed(random_seed)

    if lan == "en":
        for num in range(2, 24):
            all_list.append(get_item("{} hours".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 hour", entity_type, lan, aliases=["an hour"]))
        all_list.append(get_item("half an hour", entity_type, lan, aliases=["half hour"]))
    elif lan == "de":
        for num in range(2, 24):
            all_list.append(get_item("{} stunden".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 stunde", entity_type, lan))
        all_list.append(get_item("einen halbe stunde", entity_type, lan))
    elif lan == "fr":
        for num in range(2, 24):
            all_list.append(get_item("{} heures".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 heure", entity_type, lan))
        all_list.append(get_item("une demi heure", entity_type, lan))
    elif lan == "it":
        for num in range(2, 24):
            all_list.append(get_item("{} ore".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 ora", entity_type, lan))
        all_list.append(get_item("mezzo ora", entity_type, lan, aliases=["mezz ora"]))
    else:
        print("ERROR: Wrong language!")

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.sample(all_list, k=entity_amount)


def make_days(entity_type, lan, entity_amount=None, max_range=10, random_seed=42):
    """
    entity_amount: An integer defining the length of the returned list
    """
    all_list = []
    # random.seed(random_seed)

    if lan == "en":
        for num in range(2, max_range):
            all_list.append(get_item("{} days".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 day", entity_type, lan, aliases=["a day"]))
        all_list.append(get_item("half a day", entity_type, lan, aliases=["half day"]))
    elif lan == "de":
        for num in range(2, max_range):
            all_list.append(get_item("{} tage".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 tag", entity_type, lan))
        all_list.append(get_item("einen halben tag", entity_type, lan))
    elif lan == "fr":
        for num in range(2, max_range):
            all_list.append(get_item("{} jours".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 jour", entity_type, lan))
        all_list.append(get_item("une demi journée", entity_type, lan))
    elif lan == "it":
        for num in range(2, max_range):
            all_list.append(get_item("{} giorni".format(str(num)), entity_type, lan))
        all_list.append(get_item("1 giorno", entity_type, lan))
        all_list.append(get_item("mezzo giornata", entity_type, lan))
    else:
        print("ERROR: Wrong language!")

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.sample(all_list, k=entity_amount)


def make_positions(entity_type, lan, entity_amount=None, max_range=200, random_seed=42):
    all_list = []
    #  random.seed(random_seed)

    for num in range(1, max_range):
        all_list.append(get_item(str(num), entity_type, lan))

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.sample(all_list, k=entity_amount)


def make_timestamp_word(entity_type, lan, entity_amount=None, random_seed=42):
    all_list = []
    # random.seed(random_seed)

    if lan == "en":
        value_list = [
            'now',
            'today',
            'tomorrow',
            'the day after tomorrow',
            'yesterday',
            'the day before yesterday',
            'on monday',
            'on tuesday',
            'on wednesday',
            'on thursday',
            'on friday',
            'on saturday',
            'on sunday',
            'in the morning',
            'in the evening',
            'in the afternoon'
        ]
    elif lan == "de":
        value_list = [
            'jetzt',
            'heute',
            'Morgen',
            'übermorgen',
            'gestern',
            'Vorgestern',
            'am Montag',
            'am Dienstag',
            'am Mittwoch',
            'am Donnerstag',
            'am Freitag',
            'am Samstag',
            'am Sonntag',
            'am Morgen',
            'am Abend',
            'am Nachmittag'
        ]
    elif lan == "fr":
        value_list = [
            'maintenant',
            'aujourd hui',
            'demain',
            'le surlendemain',
            'hier',
            'avant hier',
            'le lundi',
            'mardi',
            'mercredi',
            'jeudi',
            'le vendredi',
            'le samedi',
            'le dimanche',
            'le matin',
            'dans la soirée',
            'dans l après midi'
        ]
    elif lan == "it":
        value_list = [
            'adesso',
            'oggi',
            'Domani',
            'il giorno dopo domani',
            'ieri',
            'laltro ieri',
            'di lunedi',
            'martedì',
            'di mercoledì',
            'di giovedì',
            'di venerdì',
            'di sabato',
            'di domenica',
            'di mattina',
            'in serata',
            'nel pomeriggio'
        ]
    else:
        print("ERROR: Wrong language!")
        value_list = []

    for value in value_list:
        all_list.append(get_item(value, entity_type, lan))

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.sample(all_list, k=entity_amount)


def make_timestamp_date(entity_type, lan, entity_amount=None, random_seed=42):
    all_list = []
    # random.seed(random_seed)

    if lan == "en":
        prepend = 'on '
    elif lan == 'de':
        prepend = 'am '
    else:
        prepend = ''

    for month in range(1, 13):
        for day in range(1, 32):
            all_list.append(get_item(prepend + "{}.{}".format(str(day), str(month)), entity_type, lan))

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.sample(all_list, k=entity_amount)


def make_timestamp_clock(entity_type, lan, entity_amount=None, random_seed=42, is_special=False):
    all_list = []
    special_time_list = []
    # random.seed(random_seed)

    for hour in range(0, 24):
        hour_str = str(hour)
        hour_str = hour_str if len(hour_str) > 1 else "0" + hour_str

        aliases = ["{} o clock".format(str(hour))]
        if hour > 12:
            aliases.append("{} pm".format(str(hour % 12)))
            aliases.append("{} p m".format(str(hour % 12)))
        else:
            aliases.append("{} am".format(str(hour)))
            aliases.append("{} a m".format(str(hour % 12)))
        special_time_list.append(
            get_item("{}:{}".format(hour_str, '00'), entity_type, lan, aliases=aliases))

        aliases = ["half past {}".format(str(hour))]
        special_time_list.append(
            get_item("{}:{}".format(hour_str, '30'), entity_type, lan, aliases=aliases))

        for minute in range(0, 60):
            aliases = []
            minute_str = str(minute)
            minute_str = minute_str if len(minute_str) > 1 else "0" + minute_str
            all_list.append(
                get_item("{}:{}".format(hour_str, minute_str), entity_type, lan, aliases=aliases))
            # if lan == 'en':
            #     if minute == 0:
            #         aliases.append("{} o clock".format(str(hour)))
            #         if hour > 12:
            #             aliases.append("{} pm".format(str(hour % 12)))
            #             aliases.append("{} p m".format(str(hour % 12)))
            #         else:
            #             aliases.append("{} am".format(str(hour)))
            #             aliases.append("{} a m".format(str(hour % 12)))
            #     elif minute == 30:
            #         aliases.append("half past {}".format(str(hour)))
            #     else:
            #         all_list.append(
            #             get_item("{}:{}".format(hour_str, minute_str), entity_type, lan, aliases=aliases))
            # elif lan == 'de':
            #     if minute == 0:
            #         aliases.append("{} uhr".format(str(hour)))
            #     elif minute == 30:
            #         aliases.append("halb {}".format(str(hour % 12 + 1)))
            #     else:
            #         aliases.append("{}:{} uhr".format(hour_str, minute_str))
            #         all_list.append(
            #             get_item("{}:{}".format(hour_str, minute_str), entity_type, lan, aliases=aliases))
            # else:
            #     print("ERROR: Wrong language!")
    if is_special:
        return special_time_list
    else:
        if entity_amount is None:
            entity_amount = len(all_list)
        return random.sample(all_list, entity_amount) + special_time_list


def merge_entity_types(entity_df, type_list):
    df_list = []
    for type_ in type_list:
        df_list.append(entity_df[entity_df.type == type_])
    collection_df = pd.concat(df_list)

    for type_ in type_list:
        collection_df['type'] = type_
        entity_df = pd.concat([entity_df, collection_df])
    entity_df = entity_df.drop_duplicates(subset=['value', 'type', 'language']).reset_index(drop=True)
    return entity_df
