import cv2

# Parameters
gap = 20 # Numbers of frames to be skipped after saving a frame
input_dir = '/home/ibm/F/CollabLoc-test/kelly2_model_construction/video' # Video directory
output_dir = '/home/ibm/F/CollabLoc-test/kelly2_model_construction/image_test' # Output images directory
video_idx_range = (0, 1) # Video indecies range
video_pre = 'room' # Video prefix
output_pre = 'db_' # Output images prefix
show_im = False # Set to True to visualize, or set to False to speed up


# Extract video frames
def main():
    false_cnt = 0
    for video_idx in range(video_idx_range[0], video_idx_range[1]):
        frame_idx = 0
        print(f'Processing {video_pre}{video_idx}.MP4')

        cap = cv2.VideoCapture(f'{input_dir}/{video_pre}{video_idx}.mp4')
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_idx += 1

                if frame_idx % gap == 1:
                    if show_im:
                        cv2.imshow('frame', frame)
                        cv2.waitKey(1)
                    frame_idx_str = str(frame_idx).zfill(5)
                    cv2.imwrite(f'{output_dir}/{output_pre}{video_pre}{video_idx}_{frame_idx_str}.jpg', frame)

                false_cnt = 0
            else:
                false_cnt += 1
                if false_cnt > 300:
                    cap.release()
                    break


if __name__ == '__main__':
    main()
