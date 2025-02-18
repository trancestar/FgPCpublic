import os
import shutil
import configargparse

def get_config_FgPC(config_file, result_folder = None, logger = None):

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('-c', default = config_file, is_config_file=True, help='config file path')
    p.add('--H', default = 5, type=int, help='Number of harmonics')
    p.add('--nPt', default = 1000, type=int, help='Number of evaluation points')
    p.add('--ngPC', default = 4, type=int, help='Number of poylnomial chaos degree')
    p.add('--nQPt', default = 10000, type=int, help='Number of quadrature points')
    p.add('--distStr', default = "uniform", type=str, help='Distribution of random variables')
    p.add('--low', default = 0.5, type=float, help='Lower bound of RV or mean')
    p.add('--high', default = 1.5, type=float, help='Upper bound of RV or variance')
    p.add('--sampleNr', default = 100000, type=float, help='number of Samples drawn from Distribution')
    config = p.parse_args()

    # write configs to logger
    if logger is not None: logger.info("User input (cmd > config > default): \n" + p.format_values())    

    # Copy config file to result path
    if result_folder is not None:
        config_filename = os.path.basename(config_file)
        shutil.copy(config_file, result_folder + "/" + config_filename)

    return config

def get_config_Integration(config_file, result_folder = None, logger = None):

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('-c', default = config_file, is_config_file=True, help='config file path')
    p.add('--V_0', default = -60, type=float, help='Initial value of V in µM')
    p.add('--n_0', default = 0, type=float, help='Initial value of n in µM')
    p.add('--Ca_0', default = 0.1, type=float, help='Initial value of Ca in µM')
    p.add('--t0', default = 0, type=float, help='Start time')
    p.add('--t_end_tr', default = 7, type=int, help='Transient process end time in minutes')
    p.add('--t_step_tr', default = 1000, type=float, help='Transient process time step in miliseconds')
    p.add('--t_end_ss', default = 24, type=int, help='End time in minutes')
    p.add('--t_step_ss', default = 500, type=float, help='Time step in miliseconds')
    p.add('--fftMinVal', default = 0.001, type=float, help='FFT Threshold value')
    p.add('--fftHighFreq', default = 500, type=float, help='FFT highest frequency')
    p.add('--fftDist', default = 50, type=float, help='FFT distance between peaks')
    
    config = p.parse_args()

    # write configs to logger
    if logger is not None: logger.info("User input (cmd > config > default): \n" + p.format_values())    

    # Copy config file to result path
    if result_folder is not None:
        config_filename = os.path.basename(config_file)
        shutil.copy(config_file, result_folder + "/" + config_filename)

    return config