"""cli entry point"""

import argparse

from clip_video_encode import clip_video_encode


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser()
      
    parser.add_argument('--src',type=str, help="path/to/video/name.mp4", nargs="+", required=True)
    parser.add_argument('--dest',type=str, help="path/to/numpy/embedding/name.npy", required=False, default="")
    parser.add_argument('--output_format',type=str, help="choose files or webdataset", required=False, default="files")
    parser.add_argument('--take_every_nth',type=int, help="take every nth frame", required=False, default=1)
    parser.add_argument('--frame_workers',type=int, help="number of Processes to distribute video reading to.", required=False, default=1)
    parser.add_argument('--frame_memory_size',type=int, help="GB of memory for FrameReader.", required=False, default=4)
    parser.add_argument('--metadata_columns',type=str, help="a comma separated list of metadata column names to look for in src", required=False, default="")
    parser.add_argument('--use_dst_name',action='store_true', help="use the save name suggested by video2numpy", required=False, default=False)

    args = parser.parse_args()
    # print ("args: ",args)

    clip_video_encode(src=args.src,
    dest=args.dest,
    output_format=args.output_format,
    take_every_nth=args.take_every_nth,
    frame_workers=args.frame_workers,
    frame_memory_size=args.frame_memory_size,
    metadata_columns=args.metadata_columns,
    use_dst_name=args.use_dst_name)

if __name__ == "__main__":
    main()
