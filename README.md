# Handwritten Digits Recognition

This is a Deep Learning project on Computer Vision.
    
The objective of the project is to capture a video from webcam or a saved file and recognize the handwritten digits in the video using a CNN.
    
## Requirements

Install required packages using pip:

```
pip install -r requirements.txt
```

To perform handwritten digit recognition:

```
python main.py
```

On running main.py, the prompt requires user input to decide if saved video or webcam is to be used for recognition. 

Enter:

- `0` - To use webcam
- `1` - To use the saved video in data folder

Upon providing the option, 5 windows will appear:
- The original video frames

<p align="center">
<img src=assets/main-window.jpg />
</p>

- 4 preprocessed frames of the digits in the image

<p align="center">
<img src=assets/digit-1.jpg />
</p>

<p align="center">
<img src=assets/digit-2.jpg />
</p>

<p align="center">
<img src=assets/digit-3.jpg />
</p>

<p align="center">
<img src=assets/digit-4.jpg />
</p>


Finally, to terminate the program, hit `q` on the keyboard.