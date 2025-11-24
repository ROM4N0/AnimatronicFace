# AnimatronicFace

## Intro
This project was developed during my volunteering at Komtek Örebro. The main goal was to create realistic mouth movement to inspire kids and teenagers to explore STEM. After being challenged by my mentor Helena to build the mouth mechanism, I also decided to add eye movement based on tracking motion from a webcam’s live feed.

The eye mechanism was inspired by Will Cogley, and the design used here is based on the same system you can find on his website:  
https://willcogley.notion.site/

## Code
This repository contains three main files used to control an animatronic face composed of moving eyes and a mouth. It includes 1 Arduino file and 2 Python files that use different approaches for computer vision, while using the same approach for mouth movement.

### Mouth Movement
Both Python scripts implement the same method for mouth motion. The user selects a `.wav` file, which the program plays. The volume of the audio at each moment determines how much the mouth opens or closes:

- Higher volume → wider mouth opening  
- Lower volume → smaller mouth opening  

This creates a simple but effective lip-sync effect.
The sound processing is done in a different thread to allow the eyes to be always looking (pun intended) for movement while the mouth can wait for the input to play different sounds/audio files.

### finalv3.py
This script uses a simpler approach compared to `hopefullyLast.py`. It performs frame subtraction on the webcam feed and computes the average position of detected movement to determine where the eyes should look.

Although it works well for singular movements, it has one clear limitation: if multiple movements happen on opposite sides of the frame simultaneously, they cancel out, causing the eyes to look straight ahead. Since the project was developed to be engaging for kids, this limitation motivated me to create a more reliable method.

### hopefullyLast.py
This version also uses frame subtraction, but adds a clustering layer at the end. The detected movement pixels are grouped into different clusters, allowing the system to focus on a specific movement without averaging them together.

The downside is higher computational cost. Because of this, both versions are included so the project can run smoothly on weaker computers.

### arduino.ino
This file contains a simple but essential Arduino script that receives commands from the Python programs and moves the servos controlling the eyes and mouth.

## Future work
As future improvements, I plan on updating the project to also take audio from a microphone and having the mouth moving with the live audio capture as that could be more interactive for the kids.
