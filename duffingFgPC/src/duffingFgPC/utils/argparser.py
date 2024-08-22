import os
import shutil
import configargparse

def get_config_Integration(config_file, result_folder, logger = None):

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('-c', default = config_file, is_config_file=True, help='config file path')
    p.add('--alpha', default = 1, type=float, help='linear stiffness')
    p.add('--beta', default = 1, type=float, help='nonlinear stiffness')
    p.add('--delta', default = 0.08, type=float, help='damping')
    p.add('--gamma', default = 0.2, type=float, help='force amplitude')
    p.add('--omega', default = 1.4, type=float, help='force frequency')
    p.add('--x0', default = 0.0, type=float, help='initial value')
    p.add('--xt0', default = 0.0, type=float, help='initial value derivative')
    p.add('--t0', default = 0.0, type=float, help='initial time')
    p.add('--tE', default = 100.0, type=float, help='end time')
    p.add('--dt', default = 0.1, type=float, help='time step')
    p.add('--fftMinVal', default = 0.001, type=float, help='FFT Threshold value')
    p.add('--fftHighFreq', default = 2, type=float, help='FFT highest frequency')
    p.add('--fftDist', default = 50, type=float, help='FFT distance between peaks')
    
    config = p.parse_args()


    # write configs to logger
    if logger is not None: logger.info("User input (cmd > config > default): \n" + p.format_values())    

    # Copy config file to result path
    config_filename = os.path.basename(config_file)
    shutil.copy(config_file, result_folder + "/" + config_filename)

    return config


def get_config_FgPC(config_file, result_folder, logger = None):

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('-c', default = config_file, is_config_file=True, help='config file path')
    p.add('--H', default = 3, type=int, help='Number of harmonics')
    p.add('--nPt', default = 100, type=int, help='Number of points for collocation')
    p.add('--ngPC', default = 0, type=int, help='Degree of gPC')
    p.add('--nQPt', default = 100, type=int, help='Number of points for quadrature')
    p.add('--distStr', default = "uniform", type=str, help='Distribution for gPC')
    p.add('--low', default = 0.8, type=float, help='lower bound of distribution')
    p.add('--high', default = 1.2, type=float, help='upper bound of distribution')
    p.add('--sampleNr', default = 100, type=int, help='number of samples')
    config = p.parse_args()

    # write configs to logger
    if logger is not None: logger.info("User input (cmd > config > default): \n" + p.format_values())    

    # Copy config file to result path
    config_filename = os.path.basename(config_file)
    shutil.copy(config_file, result_folder + "/" + config_filename)

    return config