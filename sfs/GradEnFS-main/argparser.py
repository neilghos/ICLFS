import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='GradEnFS main program.')
    
    # experiment setting
    parser.add_argument('--use_seeds', action='store_true', help='Choose to control random seeds for initializing network or not.')
    parser.add_argument('--seeds', type=int, action='store', nargs='*', default=[0, 1, 2, 3, 4], help='The list of seed for various repeating experiments, the seed is used to control the generation of network.')
    parser.add_argument('--repeat', type=int, default=5, help='The number of different experiment runs (default: 5).')
    parser.add_argument('--save_model', action='store_true', help='Choose to save the model checkpoint after every run.')
    parser.add_argument('--evaluation_model', type=str, default='svm', help='Choose the model for evaluation (default: svm, option: svm, knn, extratree).')

    return parser