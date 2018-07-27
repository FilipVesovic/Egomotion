import argparse

parser = argparse.ArgumentParser(description='Train and visualize Egomotion.')
parser.add_argument("train")
parser.add_argument("visualize")
args = parser.parse_args()

print(args)
