# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import lldb

def leaf_kv_meta(valobj, internal_dict):
	key = valobj.GetChildMemberWithName('key_len').GetValueAsUnsigned()
	offset = valobj.GetChildMemberWithName('offset').GetValueAsUnsigned()
	encoded = valobj.GetChildMemberWithName('value_len_op_type_ref').GetChildMemberWithName("v").GetChildMemberWithName("value").GetValue()

	return f'key_len: {key}, offset: {offset}, encoded: {encoded}'

class BaseProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj
        process = self.valobj.GetProcess()
        self._endianness = process.GetByteOrder()
        self._pointer_size = process.GetAddressByteSize()
        self._char_type = valobj.GetType().GetBasicType(lldb.eBasicTypeChar)

    def has_children(self):
        return True

    def gen_child(self, name, value):
        data = None
        type = None
        if isinstance(value, int):
            data = lldb.SBData.CreateDataFromUInt64Array(self._endianness, self._pointer_size, [value])
            type = self.valobj.target.GetBasicType(lldb.eBasicTypeLong)
        elif isinstance(value, float):
            data = lldb.SBData.CreateDataFromDoubleArray(self._endianness, self._pointer_size, [value])
            type = self.valobj.target.GetBasicType(lldb.eBasicTypeDouble)
        elif isinstance(value, str):
            data = lldb.SBData.CreateDataFromCString(self._endianness, self._pointer_size, value)
            type = self.valobj.target.GetBasicType(lldb.eBasicTypeChar).GetArrayType(len(value))

        if (data is not None) and (type is not None):
            return self.valobj.CreateValueFromData(name, data, type)
        return None


class LeafNodeProvider(BaseProvider):
	def num_children(self):
		return 4

	def has_children(self):
		return True

	def get_child_index(self, name):
		if name == 'meta':
			return 0
		if name == 'next_level':
			return 1
		if name == 'prefix_len':
			return 2
		if name == 'kv_meta':
			return 3
		return None

	def get_child_at_index(self, index):
		logger = lldb.formatters.Logger.Logger()
		if index == 0:
			return self.valobj.GetChildMemberWithName('meta')
		if index == 1:
			return self.valobj.GetChildMemberWithName('next_level')
		if index == 2:
			return self.valobj.GetChildMemberWithName('prefix_len')
		if index == 3:
			return self.valobj.GetChildMemberWithName('data')
		return None

ID_MASK = 0x4000_0000_0000_0000;
class PidProvider(BaseProvider):
    def num_children(self):
        return 3
    
    def get_child_index(self, name):
        if name == 'type':
            return 0
        if name == 'as_id':
            return 1
        if name == "as_ptr":
            return 2
        return None
    
    def get_child_at_index(self, index):
        logger = lldb.formatters.Logger.Logger()
        encoded = self.valobj.GetChildMemberWithName('value').GetValueAsUnsigned()
        if index == 0:
            is_id = encoded & ID_MASK
            return self.gen_child('type', 'id' if is_id else 'ptr')
        if index == 1:
            id_v = encoded & ~ID_MASK
            return self.gen_child('as_id', id_v) 
        if index == 2:
            return self.gen_child('ptr', encoded)
        return None

HAS_FENCE_MASK = 0x8000
SHOULD_SPLIT_MASK = 0x4000
CHILDREN_IS_LEAF_MASK = 0x2000
VALUE_COUNT_MASK = 0x1FFF

class NodeMetaProvider(BaseProvider):
	def num_children(self):
		return 6

	def get_child_index(self, name):
		if name == 'node_size':
			return 0
		if name == 'remaining_size':
			return 1
		if name == 'has_fence':
			return 2
		if name == 'need_split':
			return 3
		if name == 'children_is_leaf':
			return 4
		if name == 'meta_count':
			return 5
		return None


	def get_child_at_index(self, index):
		logger = lldb.formatters.Logger.Logger()
		if index == 0:
			return self.valobj.GetChildMemberWithName('node_size')
		if index == 1:
			return self.valobj.GetChildMemberWithName('remaining_size')

		encoded = self.valobj.GetChildMemberWithName('encoded').GetValueAsUnsigned()
		if index == 2:
			val = (encoded & HAS_FENCE_MASK) > 0
			return self.gen_child('has_fence', val)
		if index == 3:
			val = (encoded & SHOULD_SPLIT_MASK) > 0
			return self.gen_child('need_split', val)
		if index == 4:
			val = (encoded & CHILDREN_IS_LEAF_MASK) > 0
			return self.gen_child('children_is_leaf', val)
		if index == 5:
			val = encoded & VALUE_COUNT_MASK
			return self.gen_child('meta_count', val)


OP_TYPE_SHIFT = 14
VALUE_LEN_MASK = 0x1F_FF
REF_BIT_MASK = 0x20_00

class LeafKvMetaProvider(BaseProvider):
	def num_children(self):
		return 6


	def get_child_index(self, name):
		if name == 'key_len':
			return 0
		if name == 'offset':
			return 1
		if name == 'preview_bytes':
			return 2
		if name == 'type':
			return 3
		if name == 'value_len':
			return 4
		if name == 'ref':
			return 5
		return None

	def get_child_at_index(self, index):
		logger = lldb.formatters.Logger.Logger()
		if index == 0:
			return self.valobj.GetChildMemberWithName('key_len')
		if index == 1:
			return self.valobj.GetChildMemberWithName('offset')
		if index == 2:
			return self.valobj.GetChildMemberWithName('preview_bytes')

		val_raw = self.valobj.GetChildMemberWithName('value_len_op_type_ref').GetChildMemberWithName("v").GetChildMemberWithName("value")
		val = val_raw.GetValueAsUnsigned()
		if index == 3:
			val = val >> OP_TYPE_SHIFT
			op_type = ''
			if val == 0:
				op_type = 'Insert'
			elif val == 1:
				op_type = 'Delete'
			elif val == 2:
				op_type = 'Cache'
			elif val == 3:
				op_type = 'Phantom'
			return self.gen_child('type', op_type)
		if index == 4:
			val = val & VALUE_LEN_MASK
			return self.gen_child('value_len', val)
		if index == 5:
			val = val & REF_BIT_MASK
			return self.gen_child('ref', val)

		return None


	def update(self):
		pass

def mini_page_next_level(valobj, internal_dict):
	val = valobj.GetChildMemberWithName('val').GetValueAsUnsigned()
	if val == ((1<<64) - 1):
		return 'null'
	return f'{val:x}'


def __lldb_init_module(debugger, internal_dict):
    lldb.formatters.Logger._lldb_formatters_debug_level = 2
