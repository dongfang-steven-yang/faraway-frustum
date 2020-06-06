import argparse
import os


def plot_curve():

    return None


def cal_score(txt_a, txt_b, out_path):
    # todo: calculating and saving scores

    plot_curve()

    return score


def main():

    # parsing arguments
    argparser = argparse.ArgumentParser(description='Calculate Scores')
    argparser.add_argument('--txt_alpha', required=True, help='path to source txt.')
    argparser.add_argument('--txt_beta', required=True, help='path to target txt.')
    argparser.add_argument('--out_path', required=True, help='path to output')
    args = argparser.parse_args()

    # process
    done = cal_score(args.txt_alpha, args.txt_beta, args.out_path)
    if done:
        print(" - Done!")


if __name__ == '__main__':
    main()