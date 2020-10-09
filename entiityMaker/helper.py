from generator.normalizer import Normalizer
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


def make_seconds(entity_type, lan, entity_amount=None):
    all_list = []

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
    return random.choices(all_list, k=entity_amount)


def make_minutes(entity_type, lan, entity_amount=None):
    """
    entity_amount: An integer defining the length of the returned list
    """
    all_list = []
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
    return random.choices(all_list, k=entity_amount)


def make_hours(entity_type, lan, entity_amount=None):
    """
    entity_amount: An integer defining the length of the returned list
    """
    all_list = []
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
    return random.choices(all_list, k=entity_amount)


def make_days(entity_type, lan, entity_amount=None, max_range=10):
    """
    entity_amount: An integer defining the length of the returned list
    """
    all_list = []
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
    return random.choices(all_list, k=entity_amount)


def make_positions(entity_type, lan, entity_amount=None, max_range=200):
    all_list = []
    for num in range(1, max_range):
        all_list.append(get_item(str(num), entity_type, lan))

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.choices(all_list, k=entity_amount)


def make_timestamp_word(entity_type, lan, entity_amount=None):
    all_list = []
    if lan == "en":
        value_list = [
            'now',
            'today',
            'tomorrow',
            'the day after tomorrow',
            'yesterday',
            'the day before yesterday',
            'monday',
            'tuesday',
            'wednesday',
            'thursday',
            'friday',
            'saturday',
            'sunday',
            'morning',
            'evening',
            'afternoon'
        ]
    elif lan == "de":
        value_list = [
             'jetzt',
             'heute',
             'Morgen',
             'übermorgen',
             'gestern',
             'Vorgestern',
             'Montag',
             'Dienstag',
             'Mittwoch',
             'Donnerstag',
             'Freitag',
             'Samstag',
             'Sonntag',
             'Morgen',
             'Abend',
             'Nachmittag'
        ]
    elif lan == "fr":
        value_list = [
             'maintenant',
             "aujourd'hui",
             'demain',
             'le surlendemain',
             'hier',
             'avant hier',
             'Lundi',
             'Mardi',
             'Mercredi',
             'Jeudi',
             'Vendredi',
             'samedi',
             'dimanche',
             'Matin',
             'soirée',
             'après midi'
        ]
    elif lan == "it":
        value_list = [
            'adesso',
            'oggi',
            'Domani',
            'il giorno dopo domani',
            'ieri',
            "l'altro ieri",
            'Lunedi',
            'martedì',
            'mercoledì',
            'giovedi',
            'Venerdì',
            'Sabato',
            'Domenica',
            'mattina',
            'sera',
            'pomeriggio'
        ]
    else:
        print("ERROR: Wrong language!")
        value_list = []

    for value in value_list:
        all_list.append(get_item(value, entity_type, lan))

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.choices(all_list, k=entity_amount)


def make_timestamp_date(entity_type, lan, entity_amount=None):
    all_list = []
    for month in range(1, 13):
        for day in range(1, 32):
            all_list.append(get_item("{}.{}".format(str(month), str(day)), entity_type, lan))

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.choices(all_list, k=entity_amount)


def make_timestamp_clock(entity_type, lan, entity_amount=None):
    all_list = []
    for hour in range(0, 24):
        for minute in range(0, 60):
            aliases = []
            hour_str = str(hour)
            hour_str = hour_str if len(hour_str) > 1 else "0" + hour_str
            minute_str = str(minute)
            minute_str = minute_str if len(minute_str) > 1 else "0" + minute_str

            if lan == 'en':
                if minute == 0:
                    aliases.append("{} o clock".format(str(hour)))
                    if hour > 12:
                        aliases.append("{} pm".format(str(hour % 12)))
                    else:
                        aliases.append("{} am".format(str(hour)))
                if minute == 30:
                    aliases.append("half past {}".format(str(hour)))
            elif lan == 'de':
                if minute == 0:
                    aliases.append("{} uhr".format(str(hour)))
                if minute == 30:
                    aliases.append("halb {}".format(str(hour % 12 + 1)))
            else:
                print("ERROR: Wrong language!")

            all_list.append(get_item("{}:{}".format(hour_str, minute_str), entity_type, lan, aliases=aliases))

    if entity_amount is None:
        entity_amount = len(all_list)
    return random.choices(all_list, k=entity_amount)