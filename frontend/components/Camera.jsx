"use client";

import React, { useRef, useEffect } from "react";

export default function Camera({ onFramesCaptured }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Start the video stream
  useEffect(() => {
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };

    startVideo();

    return () => {
      // Stop the video stream on unmount
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Capture 10 frames from the video
  const captureFrames = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const frames = [];

    for (let i = 0; i < 10; i++) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frame = canvas.toDataURL("image/jpeg");
      frames.push(frame);
      await new Promise((resolve) => setTimeout(resolve, 100)); // Wait 100ms between captures
    }

    onFramesCaptured(frames); // Pass frames back to parent
  };

  return (
    <div>
      <video ref={videoRef} autoPlay muted className="w-full h-auto" />
      <canvas
        ref={canvasRef}
        width="320"
        height="240"
        style={{ display: "none" }}
      ></canvas>
      <button
        onClick={captureFrames}
        className="bg-blue-500 text-white px-4 py-2 rounded mt-4"
      >
        Capture Frames
      </button>
    </div>
  );
}
