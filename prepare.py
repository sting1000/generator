import pandas as pd
from pathlib import Path
import json
import warnings
from generator.module import Generator
from generator.normalizer import Normalizer
from tqdm import tqdm
warnings.filterwarnings('ignore')

# load config values
config_file = Path('config') / 'command_generator_config.json'
with open(config_file, 'r', encoding='utf-8') as fp:
    config = json.load(fp)

data_dir = Path(config['data_dir'])
output_dir = Path(config['output_dir'])
templates_filename = config['templates_filename']
entities_filename = config['entities_filename']
languages = config['languages']
id_size = config["id_size"]
extra_num_size = config['extra_num_size']

# read dataframe
templates = pd.read_json(data_dir / templates_filename)[['id'] + languages]
entities = pd.read_json(data_dir / entities_filename)

# set generator
gen = Generator(templates=templates, entities=entities, method='one')
f_source = open(output_dir / ('commands_source_' + '_'.join(languages) + '.txt'), "a")
f_target = open(output_dir / ('commands_target_' + '_'.join(languages) + '.txt'), "a")

# generate
for temp_id, size in id_size.items():
    print(temp_id)
    for _ in tqdm(range(size)):
        for lan in languages:
            command, label = gen.get_command(target_id=temp_id, target_lang=lan, verbose=False)[0]
            f_source.write("{}\n".format(' '.join(command)))
            f_target.write("{}\n".format(' '.join(label)))

print("Generating extra numbers")
for num in tqdm(range(extra_num_size)):
    for lan in languages:
        num_norm = Normalizer().normalize_text(str(num), lan)
        f_source.write("{}\n".format(str(num)))
        f_target.write("{}\n".format(num_norm))
f_source.close()
f_target.close()
