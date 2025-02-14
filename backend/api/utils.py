

def get_sort_asc_desc(model, sort, default):
    order_by = sort[0]
    filed = sort[1:]

    if hasattr(model, filed):
        sort_column = getattr(model, filed)
    else:
        sort_column = default
    if order_by == '+':
        return sort_column.asc()
    else:
        return sort_column.desc()


def get_model_by_field(field_list,field_model:dict):
    filter_field_list = list(filter(lambda x: field_model.get(x['field']) is not None, field_list))
    model_field_list = []
    #  = list(map(lambda x: x['model'] = field_model.get(x['field'])) , filter_field_list))
    for filter_field in filter_field_list:
        filter_field['model'] = field_model.get(filter_field['field'])
        model_field_list.append(filter_field)
    return model_field_list

def get_regexp_filter(x):
    if x['op'] == 'regexp':
        return True
    return False

def get_orther_filter(x):
    if (x['field'] != 'impression_str') and (x['op'] != 'regexp'):
        return True
    elif x['op'] == 'regexp':
        return False
    elif x['field'] == 'impression_str':
        return False
    return True

def get_regexp(regexp_filter_list):
    import  importlib
    module = importlib.import_module('app.model')
    regexp_list = []
    for i in regexp_filter_list:
        model = getattr(module, i['model'])
        column = getattr(model, i['field'])
        regexp_list.append(column.regexp_match(i['value'], flags='i'))
    return regexp_list