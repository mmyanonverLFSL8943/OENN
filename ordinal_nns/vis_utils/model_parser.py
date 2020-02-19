import os

def check_property(tokens, property_name):
    exists = False
    index_found = -1
    for index, element in enumerate(tokens):
        if element == property_name:
            exists = True
            index_found = index
            return exists, index_found
    return exists, index_found

def baseline_model_parser(model_path):
    experiment_name = os.path.basename(model_path)
    result_name = os.path.splitext(experiment_name)[0]
    tokens = result_name.split('_')

    if tokens[0] =='kdd_cup':
        dataset_name = tokens[0] + '_' + tokens[1]
    elif tokens[0] =='cover':
        dataset_name = tokens[0] + '_' + tokens[1]
    else:
        dataset_name = tokens[0]

    exists, index = check_property(tokens, 'model')
    assert tokens[index] == 'model'
    model_name = tokens[index +1]

    exists, index = check_property(tokens, 'layers')
    assert tokens[index] == 'layers'
    layers = int(tokens[index + 1])

    exists, index = check_property(tokens, 'dimension')
    assert tokens[index] == 'dimension'
    dimensions = int(tokens[index + 1])

    exists, index = check_property(tokens, 'hl')
    print(str(tokens[index]) + '_' + str(tokens[index+1]))
    assert str(tokens[index]) + '_' + str(tokens[index+1]) == 'hl_size'
    hl_size = int(tokens[index + 2])

    return {'dataset_name': dataset_name,
            'model_name': model_name,
            'layers': layers,
            'dimension': dimensions,
            'hl_size': hl_size}
