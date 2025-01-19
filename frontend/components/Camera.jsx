"use client";

import Image from "next/image";
import { useRef, useEffect, useState } from "react";
import axios from "axios";

const Camera = ({ setIsCamera, isCamera, isSuccess, setIsSuccess, from }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [frames, setFrames] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  // const [cameraPermission, setCameraPermission] = useState(false);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      if (videoRef.current) {
        console.log("set loading to false");
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
        };
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      setIsCamera(false);
      // setCameraPermission(false);
    }
  };

  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      // Set canvas size to match video
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      // Draw the current video frame onto the canvas
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      // Get the captured frame as a data URL (image format)
      const dataURL = canvas.toDataURL("image/png");
      // console.log("Captured Image");
      return dataURL;
    }
    return null;
  };

  const startCapture = () => {
    let capturedFrames = [];
    let captureCount = 0;

    if (!isCamera) {
      const intervalId = setInterval(() => {
        if (captureCount >= 10) {
          clearInterval(intervalId);
          if (videoRef.current) {
            videoRef.current.pause(); // Pause the video
          }

          // Save the captured frames to state
          setFrames(capturedFrames);
          setIsCapturing(false);

          // Send the frames via POST request
          sendPostRequest(capturedFrames);
          return;
        }

        setIsCapturing(true);
        const frame = captureFrame();
        if (frame) {
          capturedFrames.push(frame);
        }
        captureCount++;
      }, 1000); // Capture one frame every second
    }
  };

  const sendPostRequest = async (frames) => {
    console.log("Following frames to send:", frames);
    try {
      if (from == "setup") {
        //For registering reference endpoint
        // const response = await axios.post("https://example.com/api/register", {
        //   frames, // Sending an array of captured frames
        // });
      } else {
        //For signing in endpoint
        // const response = await axios.post("https://example.com/api/sign-in", {
        //   frames, // Sending an array of captured frames
        // });
      }

      //Fake responses for testing front-end
      const response = { status: "success" };

      if (response.status == "failed") {
        setIsCamera(false);
      } else {
        setIsSuccess(true);
      }
    } catch (error) {
      console.error("Error uploading frames:", error);
      setIsCamera(false);
    }
  };

  useEffect(() => {
    const init = async () => {
      await startCamera(); // Start the camera
      startCapture(); // Start capturing frames
    };
    init();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div>
      <video
        className="mb-2"
        ref={videoRef}
        style={{ width: "100%", height: "auto" }}
      />
      <p className="text-center text-primary mb-5">
        {isCapturing && "Capturing face..."}
      </p>
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
};

export default Camera;
