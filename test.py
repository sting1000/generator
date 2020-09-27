# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Get dataset

# %%
import pandas as pd
import numpy as np
import random
import re  
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# ## Templates

# %%
# templates
templates = pd.read_json("data/templates.json")
init_uttr_updated = pd.read_csv("data/InitUtterance_list_updated.csv")
templates_updated = templates.merge(init_uttr_updated, on='id')
templates_updated = templates_updated[['id','de', 'fr', 'it', 'en']]
templates_updated.head()


# %%
# entities
entities = pd.read_json("data/entities.json")
entities = entities[['value', 'type', 'language', 'normalizedValue', 'aliases']]
entities.head()

# %%
from generator.module import Generator, Normalizer
gen = Generator(templates=templates_updated, entities=entities)
gen.get_command(target_id='Tv.TvChannelChange.Init.Utterance', 
                target_lang='de', 
                normalizer= Normalizer().normalize_text,
                verbose=True)

# %%
target_id='MyCloud.OpenArea.Init.Utterance'
target_lang='de'
pattern_list = templates_updated[templates_updated['id'] == target_id][target_lang].values[0]['texts']
pattern = random.choice(pattern_list)['ttsText'] 
pattern
# %%
gen.remove_tags('öffne #myCloud und gehe zu {MyCloudArea}', 'de')
