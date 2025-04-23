import json
import os

def load_json_files(file_paths):
    json_data = {"horizontal": [], "vertical": []}
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            if not isinstance(data, list):
                print(f"File {file_path} does not contain a list.")
                continue
            
            if "horizontal" in file_path:
                json_data["horizontal"].extend(data)
            elif "vertical" in file_path:
                json_data["vertical"].extend(data)
            else:
                print(f"File {file_path} does not contain 'horizontal' or 'vertical' in its name.")
    
    return json_data

def concatenate_file_prefixes(file_paths, orientation):
    prefix_set = set()
    
    for file_path in file_paths:
        if orientation in file_path:
            prefix = os.path.basename(file_path).split("_")[0]
            prefix = prefix.replace(f"-human", "")
            prefix_set.add(prefix)
    
    return './data/json_files/' + "-".join(sorted(prefix_set))

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def main(file_paths):
    json_data = load_json_files(file_paths)
    
    horizontal_filename = concatenate_file_prefixes(file_paths, "horizontal") + "_horizontal.json"
    vertical_filename = concatenate_file_prefixes(file_paths, "vertical") + "_vertical.json"
    
    if json_data["horizontal"]:
        save_json(json_data["horizontal"], horizontal_filename)
        print(f"Saved horizontal data to {horizontal_filename}")
        
    if json_data["vertical"]:
        save_json(json_data["vertical"], vertical_filename)
        print(f"Saved vertical data to {vertical_filename}")

if __name__ == "__main__":
    file_paths = [
        #"data/json_files/tiktok_vertical_meta.json",
        #"data/json_files/ubc_vertical_meta.json",
        #"data/json_files/pexels_vertical.json",
        "data/json_files/pexels_horizontal.json",
        "data/json_files/ue_horizontal_meta.json"
    ]
    main(file_paths)