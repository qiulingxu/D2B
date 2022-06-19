import numpy as np
import os
import json
from shutil import copy2
from filelock import FileLock

def merge_dict(tot_dict, add):
    for k, v in add.items():
        if k in tot_dict:
            tot_dict[k] = np.concatenate([tot_dict[k], add[k]])
        else:
            tot_dict[k] = add[k]
    return tot_dict


class recorder_base:
    def __init__(self):
        self.content = None
        pass

    def _save(self, content):
        #assert False, "Please implement method"
        pass

    def _load(self,):
        #assert False, "Please implement method"
        content = None
        return content

    def _check_init_record(self, record, np_arr):
        #assert False, "Please implement method"
        return record

    def _save_record(self, record, np_arr):
        #assert False, "Please implement method"
        pass

    def _save_record_by_id(self, record, np_arr, idx):
        #assert False, "Please implement method"
        pass

    def _load_record_by_id(self, record, idx):
        #assert False, "Please implement method"
        return {}

    def _load_record_by_num(self, record, start, end):s
        #assert False, "Please implement method"
        return {}

    def _get_record_num(self, record):
        #assert False, "Please implement method"
        return 0

    def _create_record(self, name):
        return {}

    def _list_record(self):
        return []

    def _delete_key(self, record_name):
        #assert False, "Please implement method"
        pass

    def save(self,):
        self._save(self.content)

    def load(self,):
        self.content = self._load()

    def backup_to(self, path):
        """ This function is optional for backup tool"""
        pass

    def restore_from(self, path):
        """ This function is optional for backup tool"""
        pass

    def save_record_by_id(self, record_name, idx, func, overwrite=False):
        ### If record by id doesn't exist, call func and get the result to save
        ### Preloading in case of any change
        self.load()
        record = self.get_or_create_record(record_name)
        cur_num = self._get_record_num(record)
        assert idx <= cur_num, "Record number is wrong"
        if overwrite or idx == cur_num:
            np_arr = func()
            self.load()
            record = self._check_init_record(record, np_arr)
            self._save_record_by_id(record, np_arr, idx)
            self.content["Record_Name"][record_name] = record
            self.save()
        else:
            pass

    def save_record(self, record_name, np_arr):
        ### Preloading in case of any change
        self.load()
        record = self.get_or_create_record(record_name)
        record = self._check_init_record(record, np_arr)
        self._save_record(record, np_arr)
        ### Update the directory file only after changes are secure
        self.content["Record_Name"][record_name] = record
        self.save()

    def list_record(self):
        return self._list_record()

    def load_record(self, record_name):
        record = self.get_or_create_record(record_name)
        num = self._get_record_num(record)
        dict_tot = {}
        for idx in range(num):
            _dict = self._load_record_by_id(record, idx)
            dict_tot = merge_dict(dict_tot, _dict)
        return dict_tot

    def load_record_by_id(self, record_name, idx):
        record = self.get_or_create_record(record_name)
        _dict = self._load_record_by_id(record, idx)
        return _dict

    def load_record_by_num(self, record_name, start, end):
        assert 0<=start and start<=end
        record = self.get_or_create_record(record_name)
        _dict = self._load_record_by_num(record, start, end)
        return _dict

    def exists_record_by_id(self, record_name, idx):
        record = self.get_or_create_record(record_name)
        num = self._get_record_num(record)
        return idx <= num -1

    def get_or_create_record(self, record_name):
        if record_name in self.content["Record_Name"]:
            record = self.content["Record_Name"][record_name]
        else:
            record = self._create_record(record_name)
        return record

    def delete_if_exist(self, record_name):
        if record_name in self.content["Record_Name"]:
            self._delete_key(record_name)
            self.save()

class recorder(recorder_base):

    def __init__(self, config_name = "record_config", readonly=False):
        super().__init__()
        self.config_path = config_name+".json"
        self.collection_name = config_name
        self.readonly = readonly
        if os.path.exists(self.config_path):
            self.load()
            self._index_check_and_fix()
            self.save()
        else:
            self.content = {"Record_Name": {}
                        }
            self.save()

    def _index_check_and_fix(self):
        for rec_name in list(self._list_record()):
            max_idx = -1
            record = self.get_or_create_record(rec_name)
            for idx in range(self._get_record_num(record)):
                _path = self._record2path(record, idx)
                if os.path.exists(_path):
                    max_idx = idx
                else:
                    break
            if max_idx == -1:
                self._delete_key(rec_name)
            elif record["Index"] != max_idx+1:
                print("Mismatch in index of %s"%rec_name)
                self.content["Record_Name"][rec_name]["Index"] = max_idx+1

    def _delete_key(self, record_name):
        print("going to delete key %s in %s, y or n: " % (record_name, self.collection_name))
        if True:
            del(self.content["Record_Name"][record_name])
        else:
            assert False,"halt"

    def _load(self,):
        with FileLock(self.config_path+".lock"):
            with open(self.config_path, "r") as f:
                content = json.loads(f.read())
        return content

    def _save(self, content):
        if not self.readonly:
            with FileLock(self.config_path+".lock"):
                with open(self.config_path, "w") as f:
                    f.write(json.dumps(content))
                    f.flush()

    def _name2path(self, name, remote_path = None):
        if remote_path is None:
            path = os.path.join("records", self.collection_name, name)
        else:
            path = os.path.join(remote_path, "records", self.collection_name, name)
        os.makedirs(path, exist_ok = True)
        return path

    def _record2path(self, record, idx):
        path = record["Path"]
        path = os.path.join(path, "nparr%d.npy"% idx)
        return path        

    def _list_record(self):
        return list(self.content["Record_Name"].keys())

    def _create_record(self, name):

        path = self._name2path(name)
        record = {"Name" : name,
                "Path" : path,
                "Index" : 0,
                "Attributes": {},
                }
        self.content["Record_Name"][name] = record
        self.save()
        return record

    def _check_init_record(self, record, np_arr):
        if record["Index"]==0:
            for key in np_arr.keys():
                record["Attributes"][key] = {"shape": np_arr[key].shape, 
                "type": np_arr[key].dtype.name }
        else:
            if "_check" not in np_arr:
                    for key in np_arr.keys(): 
                        if key in record["Attributes"]:
                            assert list(record["Attributes"][key]["shape"][1:]) == list(np_arr[key].shape[1:]), "Shape should be matched for key %s" % key
                            assert record["Attributes"][key]["type"] == np_arr[key].dtype.name, "Data type should be matched for key %s" % key
                        else:
                            record["Attributes"][key] = {"shape": np_arr[key].shape, "type": np_arr[key].dtype.name }
        return record
    
    def _save_record(self, record, np_arr):
        idx = record["Index"]
        self._save_record_by_id(record, np_arr, idx)

    def _save_record_by_id(self, record, np_arr, idx):
        if self.readonly:
            assert False, "Read only mode does not support saving"
        ### IDX should be zero based
        cur_num = record["Index"]
        assert idx <= cur_num
        path = self._record2path(record, idx)
        np.save(path, np_arr)
        if idx == cur_num:
            record["Index"] += 1

    def _load_record_by_id(self, record, idx):
        path = self._record2path(record, idx)
        _dict = np.load(path,allow_pickle=True).item()
        return _dict

    def _load_record_by_num(self, record, start, end):
        _dicts = {}
        _sum=0
        for i in range(record["Index"]):
            _dict = self._load_record_by_id(record, i)
            l = None
            for k in _dict.keys():
                if l is None:
                    l=_dict[k].shape[0]
                else:
                    assert l == _dict[k].shape[0]
                if k in _dicts:
                    _dicts[k].append(_dict[k])
                else:
                    _dicts[k] = [_dict[k]]
            _sum += l
            if _sum>=end:
                break
            
        for k in _dicts.keys():
            _dicts[k] = np.concatenate(_dicts[k], axis=0)
        
        for k in _dicts.keys():
            assert end <= _dicts[k].shape[0], "record %s has shape %s, but requires end at %d" % (
                k, _dicts[k].shape, end)
            _dicts[k]= _dicts[k][start:end]
        return _dicts
        

    def _get_record_num(self, record):
        return record["Index"]

    def backup_to(self, path):
        remote_path = os.path.join(path, self.config_path)
        open(remote_path, "w").write(json.dumps(self.content))
        for record in self.content.keys():
            record = self.get_or_create_record(record)
            from_path = self._name2path(record["Name"])
            to_path = self._name2path(record["Name"], remote_path=path)

            for idx in range(record["Index"]):
                path1 = os.path.join(from_path, "nparr%d.npy" % idx)
                path2 = os.path.join(to_path, "nparr%d.npy" % idx)
                copy2(path1, path2)

    def restore_from(self, path):
        assert False, "Not implemented"
        pass

class recorder_memory(recorder_base):

    def __init__(self):
        super().__init__()
        self.MEM_BUFF = {}
        self.content = {"Record_Name": {}
                        }
        self.load()

    def _load(self,):
        return self.content

    def _save(self, content):
        pass

    def _name2path(self, name):
        path = ".".join(["records", name])
        return path

    def _record2path(self, record, idx):
        path = record["Path"]
        path = ".".join([path, "nparr%d" % idx])
        return path

    def _create_record(self, name):

        path = self._name2path(name)
        record = {"Name": name,
                  "Path": path,
                  "Index": 0,
                  "Attributes": {},
                  }
        self.content["Record_Name"][name] = record
        return record

    def _check_init_record(self, record, np_arr):
        if record["Index"] == 0:
            for key in np_arr.keys():
                record["Attributes"][key] = {"shape": np_arr[key].shape,
                                             "type": np_arr[key].dtype.name}
        else:
            for key in np_arr.keys():
                assert list(record["Attributes"][key]["shape"][1:]) == list(
                    np_arr[key].shape[1:]), "Shape should be matched for key %s" % key
                assert record["Attributes"][key]["type"] == np_arr[key].dtype.name, "Data type should be matched for key %s" % key
        return record

    def _save_record(self, record, np_arr):
        idx = record["Index"]
        self._save_record_by_id(record, np_arr, idx)

    def _save_record_by_id(self, record, np_arr, idx):
        ### IDX should be zero based
        cur_num = record["Index"]
        assert idx <= cur_num
        path = self._record2path(record, idx)
        self.MEM_BUFF[path] = np_arr
        if idx == cur_num:
            record["Index"] += 1

    def _load_record_by_id(self, record, idx):
        path = self._record2path(record, idx)
        _dict = self.MEM_BUFF[path]
        return _dict

    def _get_record_num(self, record):
        return record["Index"]


if __name__=="__main__":

    print("$$$$$$$$$$$$$Now testing Recorder_Disk.\n")
    def init_rc():
        return recorder("test.json")
    _rc = init_rc()

    for i in range(4):
        _arr = {"arr1": np.array((i+1)*[6])}
        _rc.save_record("test_item1", _arr)
    _rc = init_rc()

    print(_rc.load_record("test_item1"))

    print("$$$$$$$$$$$$Now testing Recorder_Memory.\n")
    def init_rc():
        return recorder_memory()

    _rc = init_rc()
    for i in range(4):
        _arr = {"arr1": np.array((i+1)*[6])}
        _rc.save_record("test_item1", _arr)

    print(_rc.load_record("test_item1"))

