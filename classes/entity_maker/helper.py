from classes.command_generator.Normalizer import Normalizer
import pandas as pd
import random


def make_item(value, item_type, lan, aliases=None):
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

def make_second(entity_type, language):
    all_list = []
    if language == "en":
        for num in range(2, 60):
            all_list.append(make_item("{} seconds".format(str(num)), entity_type, language))
        all_list.append(make_item("1 second", entity_type, language, aliases=["a second"]))
        all_list.append(make_item("half a second", entity_type, language, aliases=["half second"]))
    elif language == "de":
        for num in range(2, 60):
            all_list.append(make_item("{} sekunden".format(str(num)), entity_type, language))
        all_list.append(make_item("1 sekunde", entity_type, language))
        all_list.append(make_item("einen halbe sekunde", entity_type, language))
    elif language == "fr":
        for num in range(2, 60):
            all_list.append(make_item("{} secondes".format(str(num)), entity_type, language))
        all_list.append(make_item("1 seconde", entity_type, language))
        all_list.append(make_item("une demi seconde", entity_type, language))
    elif language == "it":
        for num in range(2, 60):
            all_list.append(make_item("{} secondi".format(str(num)), entity_type, language))
        all_list.append(make_item("1 secondo", entity_type, language))
        all_list.append(make_item("mezzo secondo", entity_type, language))
    return all_list


def make_minute(entity_type, language):
    all_list = []
    if language == "en":
        for num in range(2, 60):
            all_list.append(make_item("{} minutes".format(str(num)), entity_type, language))
        all_list.append(make_item("1 minute", entity_type, language, aliases=["a minute"]))
        all_list.append(make_item("half a minute", entity_type, language, aliases=["half minute"]))
    elif language == "de":
        for num in range(2, 60):
            all_list.append(make_item("{} minuten".format(str(num)), entity_type, language))
        all_list.append(make_item("1 minute", entity_type, language))
        all_list.append(make_item("einen halbe minute", entity_type, language))
    elif language == "fr":
        for num in range(2, 60):
            all_list.append(make_item("{} minutes".format(str(num)), entity_type, language))
        all_list.append(make_item("1 minute", entity_type, language))
        all_list.append(make_item("une demi minute", entity_type, language))
    elif language == "it":
        for num in range(2, 60):
            all_list.append(make_item("{} minuti".format(str(num)), entity_type, language))
        all_list.append(make_item("1 minuto", entity_type, language))
        all_list.append(make_item("mezzo minuto", entity_type, language))
    return all_list


def make_hour(entity_type, language):
    all_list = []

    if language == "en":
        for num in range(2, 24):
            all_list.append(make_item("{} hours".format(str(num)), entity_type, language))
        all_list.append(make_item("1 hour", entity_type, language, aliases=["an hour"]))
        all_list.append(make_item("half an hour", entity_type, language, aliases=["half hour"]))
    elif language == "de":
        for num in range(2, 24):
            all_list.append(make_item("{} stunden".format(str(num)), entity_type, language))
        all_list.append(make_item("1 stunde", entity_type, language))
        all_list.append(make_item("einen halbe stunde", entity_type, language))
    elif language == "fr":
        for num in range(2, 24):
            all_list.append(make_item("{} heures".format(str(num)), entity_type, language))
        all_list.append(make_item("1 heure", entity_type, language))
        all_list.append(make_item("une demi heure", entity_type, language))
    elif language == "it":
        for num in range(2, 24):
            all_list.append(make_item("{} ore".format(str(num)), entity_type, language))
        all_list.append(make_item("1 ora", entity_type, language))
        all_list.append(make_item("mezzo ora", entity_type, language, aliases=["mezz ora"]))
    return all_list


def make_days(entity_type, language, max_range):
    all_list = []

    if language == "en":
        for num in range(2, max_range):
            all_list.append(make_item("{} days".format(str(num)), entity_type, language))
        all_list.append(make_item("1 day", entity_type, language, aliases=["a day"]))
        all_list.append(make_item("half a day", entity_type, language, aliases=["half day"]))
    elif language == "de":
        for num in range(2, max_range):
            all_list.append(make_item("{} tage".format(str(num)), entity_type, language))
        all_list.append(make_item("1 tag", entity_type, language))
        all_list.append(make_item("einen halben tag", entity_type, language))
    elif language == "fr":
        for num in range(2, max_range):
            all_list.append(make_item("{} jours".format(str(num)), entity_type, language))
        all_list.append(make_item("1 jour", entity_type, language))
        all_list.append(make_item("une demi journée", entity_type, language))
    elif language == "it":
        for num in range(2, max_range):
            all_list.append(make_item("{} giorni".format(str(num)), entity_type, language))
        all_list.append(make_item("1 giorno", entity_type, language))
        all_list.append(make_item("mezzo giornata", entity_type, language))
    return all_list


def make_position(entity_type, language, max_range=200):
    all_list = []
    for num in range(0, max_range):
        all_list.append(make_item(str(num), entity_type, language))
    return all_list


def make_timestamp_word(entity_type, language):
    all_list = []

    if language == "en":
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
    elif language == "de":
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
    elif language == "fr":
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
    elif language == "it":
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
    for value in value_list:
        all_list.append(make_item(value, entity_type, language))
    return all_list


def make_timestamp_date(entity_type, language):
    all_list = []
    if language == "en":
        prepend = 'on '
    elif language == 'de':
        prepend = 'am '
    else:
        prepend = ''

    for month in range(1, 13):
        for day in range(1, 32):
            all_list.append(make_item(prepend + "{}.{}".format(str(day), str(month)), entity_type, language))
    return all_list


def make_timestamp_clock(entity_type, language):
    all_list = []
    for hour in range(0, 24):
        aliases = []
        hour_str = str(hour)
        hour_str = hour_str if len(hour_str) > 1 else "0" + hour_str

        # when minute == 00
        if hour > 12:
            hour_str_12 = str(hour % 12)
            aliases.append("{} pm".format(hour_str_12))
            aliases.append("{} p m".format(hour_str_12))
            aliases.append("{} o clock".format(hour_str_12))
        else:
            aliases.append("{} am".format(hour_str))
            aliases.append("{} a m".format(hour_str))
            aliases.append("{} o clock".format(hour_str))
        all_list.append(
            make_item("{}:{}".format(hour_str, '00'), entity_type, language, aliases=aliases))

        # when minute == 30
        if language == 'en':
            aliases = ["half past {}".format(str(hour))]
        elif language == 'de':
            aliases = ["halb {}".format(str(hour))]
        elif language == 'fr':
            aliases = ["{} heures et demie".format(str(hour))]
        elif language == 'it':
            aliases = ["{} e mezza".format(str(hour))]
        all_list.append(
            make_item("{}:{}".format(hour_str, '30'), entity_type, language, aliases=aliases))

        # minute is other case
        for minute in range(0, 60):
            if minute != 0 and minute != 30:
                minute_str = str(minute)
                minute_str = minute_str if len(minute_str) > 1 else "0" + minute_str
                all_list.append(
                    make_item("{}:{}".format(hour_str, minute_str), entity_type, language, aliases=[]))
        return all_list


def mix_entity_types(entity_df, type_list):
    if not type_list:
        return entity_df
    df_list = []
    for type_ in type_list:
        df_list.append(entity_df[entity_df.type == type_])
    collection_df = pd.concat(df_list)

    for type_ in type_list:
        collection_df['type'] = type_
        entity_df = pd.concat([entity_df, collection_df])
    entity_df = entity_df.drop_duplicates(subset=['value', 'type', 'language']).reset_index(drop=True)
    return entity_df
