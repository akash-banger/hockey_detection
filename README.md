# Ice Hockey Match Analysis System | Assignment | STEALTH MODE
Computer vision system for player detection, tracking, and identification in ice hockey matches.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technical Details](#technical-details)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview
This system analyzes ice hockey match videos to:
- Detect and track players on the ice
- Identify referees
- Track the puck
- Associate players with their jersey numbers and names
- Distinguish between teams

## Features
- 🏃 Real-time player and referee detection
- 🏒 Puck tracking
- 👕 Jersey number recognition
- 🎯 Team differentiation
- 📊 Player position tracking


## Technical Details

### Model Architecture
- Base Model: YOLOv11x (pre-trained)
- Three specialized fine-tuned models:
  1. Player-Referee Detection Model
  2. Puck Detection Model
  3. Jersey Number Recognition Model

### Datasets
1. **Player-Referee Dataset**
   - [Ice Hockey Dataset](https://universe.roboflow.com/ravirajsinh-dabhi-6mq2l/ice-hockey-drjvv/dataset/2)
   - Used for basic entity detection

2. **Puck Detection Dataset**
   - [Player-Puck Dataset](https://universe.roboflow.com/projects-8f38g/player-detection-b6ww5/dataset/2)
   - Enhanced detection including puck

3. **Jersey Number Dataset**
   - [Jersey Number Dataset](https://universe.roboflow.com/fastdeploy/-923m4/dataset/1)
   - Jersey number recognition



#### Jersey Number Recognition
- Two-step approach:
  - Posture detection using contour analysis
  - Number recognition using YOLO model

## Results

### Current Capabilities
✅ Accurate player and referee detection </br>
✅ Team differentiation </br>
✅ Basic player tracking </br>
✅ Jersey number detection (with limitations) </br>
⚠️ Partial puck tracking capability </br>

### Limitations
1. **Puck Detection**
   - Inconsistent tracking due to small size and rapid movement
   - Need for specialized training data

2. **Jersey Number Recognition**
   - Dependent on player orientation
   - Limited accuracy in dynamic situations

3. **Player Tracking**
   - Occasional identity switches
   - Tracking persistence issues

## Future Improvements

### Enhancements
1. **Puck Detection**
   - Implement specialized tracking algorithms
   - Increase training data variety
   - Add motion prediction

2. **Jersey Number Recognition**
   - Improve posture detection algorithm
   - Enhance model training with varied angles
   - Implement temporal consistency

3. **Player Tracking**
   - Implement robust tracking algorithms
   - Add player motion prediction
   - Improve identity persistence
