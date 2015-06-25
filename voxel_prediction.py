import sys
import argparse
import os
import rapidjson
import voxel_prediction as vp

def process_command_line():
    """Parse command line arguments.
    """
    # Add the command line arguments.
    parser = argparse.ArgumentParser(description="ilastik autocontext",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("project_file", type=str,
                        help="input file")
    parser.add_argument("-m", "--modus", type=str,
                        help="'train' or 'test' modus")

    #parser.add_argument("-n", "--nloops", type=int, default=3,
    #                    help="number of autocontext loop iterations")
    #parser.add_argument("-d", "--labeldataset", type=int, default=0,
    #                    help="id of dataset in the ilp file that contains the labels")


    parser.add_argument("-d", "--data", type=str,
                        help="data location (path)")

    parser.add_argument("-k", "--key", type=str,
                        help="key")
    
    parser.add_argument("-o", "--out", type=str,
                        help="outfile")

    parser.add_argument("-rb", "--roiBegin",nargs='+', type=int,default=None,
                        help="begin of roi")        
    parser.add_argument("-re", "--roiEnd",nargs='+', type=int,default=None,
                        help="end of roi")

    args = parser.parse_args()

    # Check arguments for validity.
    if not os.path.isfile(args.project_file):
        raise Exception("%s is not a file" % args.project_file)

    if not args.modus in ['train','test']:
        raise Exception("'%s' is an invalid modus, must be 'train' or 'test'" % args.modus)
    



    if args.modus == 'test':
        if args.data is None:
            raise Exception("--data must be set")
        if args.key is None:
            raise Exception("--key must be set")
        if not os.path.isfile(args.data):
            raise Exception("%s is not a file" % args.data)

        if args.out is None:
            raise Exception("--out must be set")
    return args


def main():
    """
    """

    # Read command line arguments.
    args = process_command_line()

    # Read the project file 
    projectFileLocation = args.project_file
    f = open(projectFileLocation,'r')
    projectFileString = f.read()
    projectFile = rapidjson.loads(projectFileString)

    if isinstance(projectFile,float):
        raise RuntimeError("could not load project-file json")
    predictor = vp.VoxelPredict(projectFile)



    if args.modus == 'train':
        predictor.doTraining()
    else :
        if args.roiBegin is None or args.roiEnd is None:
            predictor.predict(dataPath=args.data, dataKey=args.key, outPath=args.out)
        else:
            roiBegin = args.roiBegin
            roiEnd = args.roiEnd
            print("ROI ",roiEnd,roiBegin)

            predictor.predictROI(dataPath=args.data, dataKey=args.key, outPath=args.out,
                roiBegin=roiBegin,roiEnd=roiEnd )
    return 0

if __name__ == "__main__":
    status = main()
    sys.exit(status)
