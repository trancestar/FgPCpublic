import os
from datetime import datetime, date
import shutil

def all_subdirs_of(b='.'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd): result.append(d)
    return result

def create_result_folder(tag = "", max_res_folders = None, max_daily_folders = None, archive = False):

    tag = tag.replace(" ", "_")

    today = date.today()

    # dd/mm/YY
    d1 = today.strftime("%Y-%m-%d")

    # Folder path
    if archive:
        path = "results/Archive/" + d1
    else:
        path = "results/" + d1

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

    # Keep only last 7 daily result folders
    if not archive:
        if max_daily_folders is not None:
            date_format = '%Y-%m-%d'
            remove_folders("./results", max_daily_folders, date_format)

    # Output Folder
    now = datetime.now()
    time_string = now.strftime("%H_%M_%S")
    target_folder_path = path + '/' + time_string + "_" + tag 

    # Remove Output folders greater than 
    if not archive:
        if max_res_folders is not None:
            date_format = '%H_%M_%S'
            remove_folders(path, max_res_folders, date_format)

    # Create target folder
    os.makedirs(target_folder_path)

    return target_folder_path

def remove_folders(path, max_keep, date_format):
    
        folders = all_subdirs_of(path)
        date_obj = []
        if date_format == '%Y-%m-%d':
            max_idx = 10
        elif date_format == '%H_%M_%S':
            max_idx = 8

        for day_folder in folders:
            
            try:
                date_obj.append(datetime.strptime(day_folder[0:max_idx], date_format))
            except ValueError:
                if day_folder != "Archive":
                    print(f'Could not interprete foldername: "{day_folder}" . Skiped folder')

        remove_staps = []

        if len(date_obj) > max_keep:
            n_remove = len(date_obj) - max_keep
            date_obj.sort()

            for i in range(n_remove):
                remove_staps.append(date_obj[i].strftime(date_format))

            for day_folder in folders:
                for remove_stap in remove_staps:
                    if day_folder[0:max_idx] == remove_stap:
                        shutil.rmtree(path + "/" + day_folder)

def save_script(filepath, tag = "", max_daily_folders = None, max_res_folders = None, archive = False):
    target_folder_path = create_result_folder(tag, max_res_folders, max_daily_folders, archive)

    # Create target file
    target_file = target_folder_path + "/executed_file_" + tag + ".py"

    # Copy script
    shutil.copy(filepath, target_file)

    return target_folder_path