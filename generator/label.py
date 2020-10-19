from generator.Derivative import Derivative
from generator.normalizer import Normalizer
import re

def add_node_to_path(derivatives, paths):
    res = []
    for p in paths:
        res.append(derivatives + p)
    return res


def get_rewrite_option_list(param):
    # TODO
    pass


class LabelMaker:
    def __init__(self, command: str, tags: str, lang: str):
        self.command = command
        self.tags = tags
        self.lang = lang
        self.command_list = self.command.split()
        self.tags_list = self.tags.split()
        self.normalizer = Normalizer().normalize_text

    def make_derivatives(self) -> list:
        # TODO: sanity check command
        if len(self.command) != self.tags_list:
            print("Cannot match command and tags: ", self.command, self.tags)
            return []
        pos = 0
        derivatives = self.search(pos)
        return derivatives

    def search(self, pos) -> list:
        if pos == len(self.command_list):
            return []
        paths_curr = []
        paths_prev = self.search(pos + 1)
        derivatives_list = self.apply_strategy(pos)
        for derivatives in derivatives_list:
            paths_curr += add_node_to_path(derivatives, paths_prev)
        return paths_curr

    def is_category(self, pos: int, category: str) -> bool:
        token = self.command_list[pos]
        tag = self.tags_list[pos]
        tag_flag = False
        if category == 'word':
            regex = r'^[a-zA-Z]+$'
            tag_flag = True
        elif category == 'abbreviation':
            regex = r'^[A-Z]+$'
            tag_flag = True
        elif category == 'time':
            regex = r'^[0-9]+[\:[0-9]+]?$'
            if tag in ['LocalsearchTimeStampStartTime', 'LocalsearchTimeStampEndTime']:
                tag_flag = True
            else:
                tag_flag = False
        elif category == 'number':
            regex = r'^[0-9]+[\.[0-9]+]?$'
        elif category == 'date':
            regex = r'^[0-9]+\.[0-9]+$'
            if tag in ['LocalsearchTimeStampStartDay']:
                tag_flag = True
            else:
                tag_flag = False
        else:
            regex = r''

        if re.findall(regex, token) and tag_flag:
            return True
        else:
            return False

    def apply_strategy(self, pos) -> list:  # list of Derivative list
        derivatives = []
        if self.is_category(pos, 'word'):
            derivatives += self.rewrite_word(self.command_list[pos])
        elif self.is_category(pos, 'abbreviation'):  # [A-Z]+
            derivatives += self.rewrite_abbr(self.command_list[pos])
        elif self.is_category(pos, 'time'):  # [0-9]+[\:[0-9]+]? && tag = timstampTime
            derivatives += self.rewrite_time(self.command_list[pos])
        elif self.is_category(pos, 'number'):  # [0-9]+
            derivatives += self.rewrite_number(self.command_list[pos])
        elif self.is_category(pos, 'date'):  # 13.3
            derivatives += self.rewrite_date(self.command_list[pos])
        else:
            pass
        return derivatives

    def rewrite_date(self, date_str):
        # TODO: append th to ordinal number
        current_derivative = []
        date_norm = self.normalizer(date_str, self.lang)
        day, month = date_norm.split()[:-1], date_norm.split()[-1]
        rewrite_list = get_rewrite_option_list(day)
        for ind in range(len(day)):
            if ind == 0:
                der = Derivative(token=rewrite_list[ind], rewrite=rewrite_list[ind], space=None, append=None)
            else:
                der = Derivative(token=rewrite_list[ind], rewrite=rewrite_list[ind], space='No', append=None)
            current_derivative.append(der)
        current_derivative.append(Derivative(token=month, rewrite=None, space='No', append=None))
        return [current_derivative]

    def rewrite_word(self, word_str):
        # TODO: Lower case for all or not?
        derivatives = [[Derivative(token=word_str, rewrite=None, space=None, append=None)]]
        # for word not all lower cased, like "Schweizer"
        if not word_str.islower():
            derivatives.append(
                [Derivative(token=word_str.lower(), rewrite=None, space=None, append=None)])
        return derivatives

    def rewrite_abbr(self, abbr_str):
        derivatives = []
        # e.g. SRF & srf
        derivatives.append([Derivative(token=abbr_str, rewrite=None, space=None, append=None)])
        derivatives.append([Derivative(token=abbr_str.lower(), rewrite=None, space=None, append=None)])
        # S R F & s r f
        derivative_upper = []
        derivative_lower = []
        for ind in range(len(abbr_str)):
            if ind == 0:
                derivative_upper.append(Derivative(token=abbr_str[ind], rewrite=None, space='No', append=None))
                derivative_lower.append(Derivative(token=abbr_str[ind].lower(), rewrite=None, space='No', append=None))
            else:
                derivative_upper.append(Derivative(token=abbr_str[ind], rewrite=None, space=None, append=None))
                derivative_lower.append(Derivative(token=abbr_str[ind].lower(), rewrite=None, space=None, append=None))
        derivatives.append(derivative_upper)
        derivatives.append(derivative_lower)
        return derivatives

    def rewrite_time(self, time_str) -> list:
        time_list = time_str.split(':')
        current_derivative = []

        if len(time_list) == 1:
            hour = time_list[0]
            current_derivative += self.rewrite_number(hour)
        else:
            hour, minutes = time_list
            current_derivative += self.rewrite_number(hour)
            current_derivative[-1].change_append('colon')
            current_derivative += self.rewrite_number(minutes)
        return [current_derivative]

    def rewrite_number(self, number_str) -> list:
        current_derivative = []
        number_norm = self.normalizer(number_str, self.lang)
        rewrite_list = get_rewrite_option_list(number_norm.split())
        for ind in range(len(number_norm)):
            if ind == 0:
                der = Derivative(token=number_norm[ind], rewrite=rewrite_list[ind], space=None, append=None)
            else:
                der = Derivative(token=number_norm[ind], rewrite=rewrite_list[ind], space='No', append=None)
            current_derivative.append(der)
        return [current_derivative]



