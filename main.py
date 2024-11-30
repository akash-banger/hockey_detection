from utils.video_utils import get_video_frames, save_video_from_frames
from trackers.trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
import cv2

def main():
    frames = get_video_frames('input_videos/input_video.mp4')
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/stub_tracks.pkl')
    
    
    
     # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    
    # for track_id, player in tracks["players"][0].items():
    #     bbox = player["bbox"]
    #     frame = frames[0]
        
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    #     cv2.imwrite(f'output_videos/player_{track_id}_cropped.jpg', cropped_image)
    #     break
    
    
    
    # output_video_frames = tracker.draw_annotations(frames, tracks)
    # save_video_from_frames(output_video_frames, 'output_videos/output_video.mp4')
    
    # print(tracks)

if __name__ == '__main__':
    main()