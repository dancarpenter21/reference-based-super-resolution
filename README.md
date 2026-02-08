# Reference-Based Super-Resolution

The use case for this web app is when the user has a copmlete low-resolution video of a scene, and a high-resolution video of the same scene that is missing frames, and wants to use the high-resolution video to train a model to upscale the complete low-resolution video.

## The Problem

Two videos of the same scene are provided, one low-resolution and one high-resolution. The high resolution video is missing frames, typically in segments a couple seconds long at a time. The low-resoution video is complete, but obviously low-resolution. The goal is to train a model to upscale the low-resolution video to high-resolution, using the high-resolution video as a reference so that the upscaled video is the same resolution as the high-resolution reference video, but has all the frames that the low-resolution video has.

## The Solution