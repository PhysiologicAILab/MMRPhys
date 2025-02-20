import pickle
import numpy as np
import argparse
from pathlib import Path

class MergeData(object):
    def __init__(self, path1, path2, dest):
        self.path1 = Path(path1)
        self.path2 = Path(path2)
        self.dest = Path(dest)

        self.big_files = sorted(list(self.path1.rglob("*input*.npy")))
        self.small_files = sorted(list(self.path2.rglob("*input*.npy")))
        assert len(self.big_files) == len(self.small_files)


    def merge_all(self):

        for i in range(len(self.small_files)):
            big_data = np.load(str(self.big_files[i]))
            small_data = np.load(str(self.small_files[i]))

            frames_dict = dict()
            frames_dict[0] = big_data
            frames_dict[1] = small_data

            save_pth = self.dest.joinpath(self.big_files[i].name)

            with open(save_pth, 'wb') as handle:  # save out frame dict pickle file
                pickle.dump(frames_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--path1', default="/mnt/sdc1/rppg/BP4D/BP4D_RGBT_180_72x72",
    #                     dest="path1", type=str, help='path for data-1, e.g. with 180x72x72 res')
    # parser.add_argument('--path2', default="/mnt/sdc1/rppg/BP4D/BP4D_RGBT_180_9x9",
    #                     dest="path2", type=str, help='path for data-2, e.g. with 180x9x9 res')
    # parser.add_argument('--dest', default="/mnt/sdc1/rppg/BP4D/BP4D_RGBT_180",
    #                     dest="dest", type=str, help='path store the merged the data')

    parser.add_argument('--path1', default="/mnt/sdc1/rppg/SCAMPS/SCAMPS_Raw_180_72x72",
                        dest="path1", type=str, help='path for data-1, e.g. with 180x72x72 res')
    parser.add_argument('--path2', default="/mnt/sdc1/rppg/SCAMPS/SCAMPS_Raw_180_9x9",
                        dest="path2", type=str, help='path for data-2, e.g. with 180x9x9 res')
    parser.add_argument('--dest', default="/mnt/sdc1/rppg/SCAMPS/SCAMPS_Raw_180",
                        dest="dest", type=str, help='path store the merged the data')

    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    mergeObj = MergeData(path1=args_parser.path1, path2=args_parser.path2, dest=args_parser.dest)

    mergeObj.merge_all()