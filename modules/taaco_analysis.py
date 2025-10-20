import os
from pathlib import Path
from TAACOnoGUI import runTAACO

initial_path = os.getcwd()
path_to_TAACO = Path(initial_path).parent.joinpath('TAACO')

def taaco_on_folder_of_files(folder_name, output_csv_name, sampleVars = {"sourceKeyOverlap" : False, "sourceLSA" : False, "sourceLDA" : False, "sourceWord2vec" : False, "wordsAll" : True, "wordsContent" : True, "wordsFunction" : True, "wordsNoun" : True, "wordsPronoun" : True, "wordsArgument" : True, "wordsVerb" : True, "wordsAdjective" : True, "wordsAdverb" : True, "overlapSentence" : False, "overlapParagraph" : False, "overlapAdjacent" : False, "overlapAdjacent2" : False, "otherTTR" : True, "otherConnectives" : True, "otherGivenness" : True, "overlapLSA" : False, "overlapLDA" : False, "overlapWord2vec" : False, "overlapSynonym" : False, "overlapNgrams" : False, "outputTagged" : False, "outputDiagnostic" : False}):
    # import folder to TAACO folder for processing
    if not os.path.exists(Path(path_to_TAACO).joinpath(folder_name)):
        os.mkdir(Path(path_to_TAACO).joinpath(folder_name))
    for file in os.listdir(Path(initial_path).joinpath(folder_name)):
        os.replace(Path(initial_path).joinpath(folder_name).joinpath(file), Path(path_to_TAACO).joinpath(folder_name).joinpath(file))
    
    # change working directory and run TAACO
    os.chdir(path_to_TAACO)
    csv_file_name = f'{output_csv_name}.csv'
    runTAACO(f'{folder_name}/', csv_file_name, sampleVars)

    # move folder and output back to main repo
    for file in os.listdir(Path(path_to_TAACO).joinpath(folder_name)):
        os.replace(Path(path_to_TAACO).joinpath(folder_name).joinpath(file), Path(initial_path).joinpath(folder_name).joinpath(file))
    Path(path_to_TAACO).joinpath(folder_name).rmdir()
    os.replace(Path(path_to_TAACO).joinpath(csv_file_name), Path(initial_path).joinpath(csv_file_name))
    
    # return to original cwd
    os.chdir(initial_path)