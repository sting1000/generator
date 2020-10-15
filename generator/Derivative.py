class Derivative:
    def __init__(self, token, rewrite=None, space=None, append=None):
        self.token = token
        self.rewrite = rewrite
        self.space = space
        self.append = append

    def change_append(self, append):
        self.append = append
