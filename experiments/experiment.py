import json
from datetime import datetime 
import os
from pathlib import Path
experiment_suffix = ".exp"
class Experiment:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get("name", "Experiment 01")
        self.time = kwargs.get("time", datetime.now())
        self.description = kwargs.get("description", "Blank Experiment")
        self.participants = kwargs.get("participants", ["subject01", "subject02"])
        self.data_folder = kwargs.get("data_folder", os.getcwd())
        self.filepath = kwargs.get("filepath", None)
        
        if self.filepath is not None:
            self.read_json(self.filepath)
        pass

    def __repr__(self):
        return f"Experiment : {self.name}\n time : {self.time}\n description : {self.description}\n participants : {self.participants}, data_folder : {self.data_folder}\n filepath : {self.filepath}"

    
    def read_json(self, filepath):# read from file
        with open(filepath, 'r') as file:
            data = json.load(file)

        self.name           = data["name"]        
        self.description    = data["description"]
        self.participants   = data["participants"]
        self.time           = datetime.strptime(data["time"], '%Y-%m-%d %H:%M:%S')
        self.data_folder    = data["data_folder"]
        self.filepath       = data["filepath"]

        
    def write_json(self, filepath):# write to json
        json_object = {
            "name"           : self.name,
            "time"           : self.time.strftime('%Y-%m-%d %H:%M:%S'),
            "description"    : self.description,
            "participants"   : self.participants,
            "data_folder"    : self.data_folder,
            "filepath"       : self.filepath
        }

        with open(filepath, 'w') as file:
            json.dump(json_object, file, indent=4)

def load_experiments(folder_path):
    experiments = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        
        if Path(entry_path).suffix == experiment_suffix:
            experiments.append(Experiment(filepath=entry_path))
    return experiments

