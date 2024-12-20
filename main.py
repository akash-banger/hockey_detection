from utils.video_utils import get_video_frames, save_video_from_frames
from trackers.trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
from metadata import players_metadata
import pickle
import os
def main():
    puck_model_path = "models/with_puck_detection.pt"
    player_referee_model_path = "models/player_referee_model.pt"
    
    players_dict = {}
    for team in players_metadata['teams'].values():
        for jersey_number, player_info in team['players'].items():
            players_dict[jersey_number] = player_info['name']
    
    
    # Get all jersey numbers from metadata
    all_jersey_numbers = list(players_dict.keys())

    frames = get_video_frames('input_videos/input_video.mp4')
    tracker = Tracker(player_referee_model_path, is_puck_detection=False)
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/stub_tracks_2.pkl')
    team_assigner = TeamAssigner()
    
    if os.path.exists('stubs/tracks_with_teams_2.pkl'):
        with open('stubs/tracks_with_teams_2.pkl', 'rb') as f:
            tracks = pickle.load(f)
    else:
        # Assign Player Teams
        # Try different frames to find one with sufficient players
        candidate_frames = [0, 60, 90, 120, 150]  # Try frames at different timestamps
        team_assignment_success = False
        
        for frame_idx in candidate_frames:
            if frame_idx >= len(tracks['players']):
                continue
                
            if len(tracks['players'][frame_idx]) > 3:  # Requiring more players for better team assignment
                team_assigner.assign_team_color(frames[frame_idx], 
                                              tracks['players'][frame_idx])
                team_assignment_success = True
                print(f"Team colors assigned using frame {frame_idx}")
                break
        
        if not team_assignment_success:
            print("Could not find enough players in any candidate frame")
            team_assigner.team_colors = {1: [0, 0, 0], 2: [0, 0, 0]}
        
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
        # Save tracks after team assignment
        with open('stubs/tracks_with_teams_2.pkl', 'wb') as f:
            pickle.dump(tracks, f)
    
    if os.path.exists('stubs/tracks_with_teams_and_jerseys_2.pkl'):
        with open('stubs/tracks_with_teams_and_jerseys_2.pkl', 'rb') as f:
            tracks = pickle.load(f)
    else:
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                jersey_number = team_assigner.get_player_jersey_number(frames[frame_num],
                                                                    track['bbox'],
                                                                    all_jersey_numbers)
                
                if jersey_number is not None:   
                    tracks['players'][frame_num][player_id]['jersey_number'] = str(jersey_number)
                    tracks['players'][frame_num][player_id]['name'] = players_dict[str(jersey_number)]
    
        # Save tracks after jersey number assignment
        with open('stubs/tracks_with_teams_and_jerseys_2.pkl', 'wb') as f:
            pickle.dump(tracks, f)
    
    output_video_frames = tracker.draw_annotations(frames, tracks)
    save_video_from_frames(output_video_frames, 'output_videos/output_video_2.mp4')

if __name__ == '__main__':
    main()