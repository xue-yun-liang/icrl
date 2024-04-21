import yaml

# read yaml file
with open('config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)


print(config_data['constraints']['normal'])
