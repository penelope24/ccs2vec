"""提供一些针对数据处理的，在全局共享的操作方法"""


def get_all_fields(max_walk_times, single_source=False):
    """
    根据dot文件中的字段生成field names
    """
    fields = list()
    fields.append("method_name")
    if not single_source:
        for i in range(max_walk_times):
            fields.append("jimple_" + str(i))
            fields.append("ir_" + str(i))
            fields.append("trans_" + str(i))
        fields.append("comment")
    else:
        for i in range(max_walk_times):
            fields.append("walk_" + str(i))
    return fields


def get_all_walk_fields(max_walk_times, single_source=False):
    """
    根据dot文件中的字段生成field names
    只生成跟walk time有关的field names
    """
    fields = list()
    if not single_source:
        for i in range(max_walk_times):
            fields.append("jimple_" + str(i))
            fields.append("ir_" + str(i))
            fields.append("trans_" + str(i))
    else:
        for i in range(max_walk_times):
            fields.append("walk_" + str(i))
    return fields