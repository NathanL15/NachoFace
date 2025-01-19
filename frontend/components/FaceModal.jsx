"use client";

import Image from "next/image";
import React, { useEffect, useState } from "react";
import Camera from "./Camera";
import Link from "next/link";
import { RiExternalLinkLine } from "@remixicon/react";

function FaceModal({ handleModalClose, from }) {
  // const [detectText, setDetectText] = useState(
  //   "Hold tight! Detecting your face...."
  // );

  // const [isLoading, setIsLoading] = useState(true);

  const [isCamera, setIsCamera] = useState(true);
  const [isSuccess, setIsSuccess] = useState(false);

  return (
    <div className="w-11/12 max-w-lg flex flex-col items-center px-8 py-12 rounded-md shadow-md bg-white absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
      <div className="mb-10 text-center">
        <h3 className="heading-xs mb-2">
          {isSuccess
            ? "Success!"
            : isCamera
            ? "Look directly at the camera"
            : "Face detection unsuccessful"}
        </h3>

        {!isCamera && (
          <Link href="/" className="text-primary flex justify-center">
            View troubleshooting tips <RiExternalLinkLine />
          </Link>
        )}
      </div>

      {isSuccess ? (
        <Image
          src="/checkmark.svg"
          alt="Checkmark"
          className="mb-10"
          width={100}
          height={100}
        />
      ) : isCamera ? (
        <Camera
          setIsCamera={setIsCamera}
          camera={isCamera}
          setIsSuccess={setIsSuccess}
          isSuccess={setIsSuccess}
          from={from}
        />
      ) : (
        <>
          <Image
            src="/detect-face-failed.svg"
            width={100}
            height={100}
            alt="failed to face"
            className="mb-10"
          />
        </>
      )}
      <button
        onClick={handleModalClose}
        className="w-full border-primary border py-3 text-primary"
      >
        Cancel
      </button>
    </div>
  );
}

export default FaceModal;
