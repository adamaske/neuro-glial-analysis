import json
import os

def load_info(filename="analysis_info.json"):
    """Loads analysis information from the JSON file."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}. Creating a new file.")
            data = {
                "regions_of_interest": {},
                "subject_ids": [],
                "events": {}
            }
            save_info(data, filename)
            return data
    else:
        data = {
            "regions_of_interest": {},
            "subject_ids": [],
            "events": {}
        }
        save_info(data, filename)
        return data

def save_info(data, filename="analysis_info.json"):
    """Saves analysis information to the JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def get_info(filename="analysis_info.json"):
    """Returns the analysis information."""
    return load_info(filename)

def set_info(new_data, filename="analysis_info.json"):
    """Sets the analysis information and saves it to the JSON file."""
    save_info(new_data, filename)

def add_subject_id(subject_id, filename="analysis_info.json"):
    """Adds a subject ID to the analysis information."""
    data = load_info(filename)
    if "subject_ids" not in data:
        data["subject_ids"] = []
    if subject_id not in data["subject_ids"]:
        data["subject_ids"].append(subject_id)
        save_info(data, filename)

def add_event(event_id, event_name, filename="analysis_info.json"):
    """Adds an event to the analysis information."""
    data = load_info(filename)
    if "events" not in data:
        data["events"] = {}
    data["events"][str(event_id)] = event_name
    save_info(data, filename)

def add_roi(roi_name, roi_data, filename="analysis_info.json"):
    """Adds a region of interest to the analysis information."""
    data = load_info(filename)
    if "regions_of_interest" not in data:
        data["regions_of_interest"] = {}
    data["regions_of_interest"][roi_name] = roi_data
    save_info(data, filename)

# Example Usage:
if __name__ == "__main__":
    filename = "analysis_info.json"

    add_subject_id(1, filename)
    add_subject_id(2, filename)
    add_subject_id(3, filename)

    add_event(1, "Pronation", filename)
    add_event(2, "Supination", filename)

    add_roi("PMA", {"LH": ["S1_D1"], "RH": ["S1_D2"]}, filename)
    add_roi("SMA", {"LH": ["S2_D2"], "RH": ["S2_D3"]}, filename)

    info = get_info(filename)
    print(json.dumps(info, indent=4))