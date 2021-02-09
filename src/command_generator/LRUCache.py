from collections import OrderedDict


class LRUCache:

    def __init__(self, size):
        self.size = size
        self.linked_map = OrderedDict()

    def set(self, key, value):
        if key in self.linked_map:
            self.linked_map.pop(key)

        if self.size == len(self.linked_map):
            self.linked_map.popitem(last=False)
        self.linked_map.update({key: value})

    def get(self, key):
        if key in self.linked_map:
            value = self.linked_map.get(key)
            self.linked_map.pop(key)
            self.linked_map.update({key: value})
            return value
        else:
            return None
