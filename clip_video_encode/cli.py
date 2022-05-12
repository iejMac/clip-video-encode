"""cli entry point"""

import fire

from clip_video_encode import clip_video_encode


def main():
    """Main entry point"""
    fire.Fire(clip_video_encode)


if __name__ == "__main__":
    main()
