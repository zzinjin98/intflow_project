import yaml


# Yaml 읽기
#script_path = os.path.dirname(__file__)
#os.chdir(script_path)
with open('C:/Users/_Admin_/PycharmProjects/intflow_project/intflow_project/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

print(config['det_path'])