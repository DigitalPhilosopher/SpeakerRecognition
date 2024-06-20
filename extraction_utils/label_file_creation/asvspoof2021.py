from typing import List, Tuple, Dict
import os
import glob

def create_label_file_asvspoof2021(data_path_dict:Dict[str, str]):
    if "asv_spoof_2021_la_eval" not in data_path_dict or (not os.path.exists(data_path_dict["asv_spoof_2021_la_eval"])):
        return
    asvspoof2021_base_path = data_path_dict["asv_spoof_2021_la_eval"]
    lines: List[str] = []
    with open(os.path.join(asvspoof2021_base_path, 'trial_metadata.txt'), 'r') as file:
        # Read each line in the file one by one
        for line in file:
            lines.append(line.strip())

    method_types = list(set([line.split(" ")[-1] for line in lines]))
    for method_type in method_types:
        method_type_name = method_type.split("_")[0]
        method_lines_list = []
        for line in lines:
            if method_type_name in line.split(" ")[-1]:
                label = line.split(" ")[-3]
                audio_path = os.path.join("flac", line.split(" ")[1] + ".flac")
                method = line.split(" ")[-4] + "_" + line.split(" ")[-5] + "_" + line.split(" ")[-6]
                method_lines_list.append(f"{audio_path},{method},{label}")
        #save method_lines_list to f"labels_{method_type}.txt"
        with open(os.path.join(asvspoof2021_base_path, f"labels_{method_type}.txt"), 'w') as file:
            for item in method_lines_list:
                file.write("%s\n" % item)



    method_types = list(set([line.split(" ")[-1] + "_" + line.split(" ")[-6] + "_" + line.split(" ")[-5] for line in lines]))
    for method_type in method_types:
        method_type_name = method_type.split("_")[0]
        method_lines_list = []
        for line in lines:
            if method_type_name in line.split(" ")[-1] and method_type.split("_")[1] in line.split(" ")[-6] and method_type.split("_")[2] in line.split(" ")[-5]:
                label = line.split(" ")[-3]
                audio_path = os.path.join("flac", line.split(" ")[1] + ".flac")
                method = line.split(" ")[-4] + "_" + line.split(" ")[-5] + "_" + line.split(" ")[-6]
                method_lines_list.append(f"{audio_path},{method},{label}")
        #save method_lines_list to f"labels_{method_type}.txt"
        with open(os.path.join(asvspoof2021_base_path,f"labels_{method_type}.txt"), 'w') as file:
            for item in method_lines_list:
                file.write("%s\n" % item)


    method_types = list(set([line.split(" ")[-1] + "_" + line.split(" ")[-4] for line in lines]))
    for method_type in method_types:
        method_type_name = method_type.split("_")[0]
        method_lines_list = []
        for line in lines:
            if method_type_name in line.split(" ")[-1] and method_type.split("_")[1] in line.split(" ")[-4]:
                label = line.split(" ")[-3]
                audio_path = os.path.join("flac", line.split(" ")[1] + ".flac")
                method = line.split(" ")[-4] + "_" + line.split(" ")[-5] + "_" + line.split(" ")[-6]
                method_lines_list.append(f"{audio_path},{method},{label}")
        #save method_lines_list to f"labels_{method_type}.txt"
        with open(os.path.join(asvspoof2021_base_path,f"labels_{method_type}.txt"), 'w') as file:
            for item in method_lines_list:
                file.write("%s\n" % item)


    #bonafide variants
    method_types = list(set([line.split(" ")[-1] + "_" + line.split(" ")[-6] + "_" + line.split(" ")[-5] for line in lines if line.split(" ")[-4] == "bonafide"]))
    for method_type in method_types:
        method_type_name = method_type.split("_")[0]
        method_lines_list = []
        for line in lines:
            if line.split(" ")[-4] != "bonafide" or (method_type_name in line.split(" ")[-1] and method_type.split("_")[1] in line.split(" ")[-6] and method_type.split("_")[2] in line.split(" ")[-5]):
                label = line.split(" ")[-3]
                audio_path = os.path.join("flac", line.split(" ")[1] + ".flac")
                method = line.split(" ")[-4] + "_" + line.split(" ")[-5] + "_" + line.split(" ")[-6]
                method_lines_list.append(f"{audio_path},{method},{label}")
        #save method_lines_list to f"labels_{method_type}.txt"
        method_name = method_type.split("_")[0] + "_bonafide_" + "_".join(method_type.split("_")[1:])
        with open(os.path.join(asvspoof2021_base_path,f"labels_{method_name}.txt"), 'w') as file:
            for item in method_lines_list:
                file.write("%s\n" % item)

