"""
(c) Copyright 2020 Swisscom AG
All Rights Reserved.
"""

# Taken from https://git.swisscom.com/projects/EELV/repos/transformer-model/browse/sanity?at=refs%2Fheads%2Fdevelop

valid_seperate_tag_names = ['c', 'overlap']
valid_tag_names = ['noise', 'vonoise', 'laugh', 'hes', '?', 'conversation', 'sil'] + valid_seperate_tag_names
valid_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜ1234567890 '
