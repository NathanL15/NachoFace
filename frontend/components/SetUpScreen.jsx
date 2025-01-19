"use client";

import {
  RiArrowLeftSLine,
  RiArrowRightSLine,
  RiExternalLinkLine,
} from "@remixicon/react";
import React, { useState } from "react";
import FaceModal from "./FaceModal";

function SetUpScreen() {
  const [showCTA, setShowCTA] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  const handleCTA = () => {
    setShowCTA(!showCTA);
  };

  const handleModalOpen = () => {
    setModalOpen(true);
  };

  const handleModalClose = () => {
    setModalOpen(false);
  };

  return (
    <div className="container py-10">
      <div className="md:flex gap-20">
        <div className="vertical-tabs flex flex-col gap-2 mb-10 md:mb-0">
          <div className="vertical-tab bg-gray-200 border-l-4 border-primary">
            Profile and Preferences
          </div>
          <div className="vertical-tab bg-gray-200">Account Services</div>
          <div className="vertical-tab bg-gray-200">
            Pay Bills and Transfer Funds
          </div>
          <div className="vertical-tab bg-gray-200">Alert Centre</div>
        </div>
        <div className="flex-1">
          {showCTA ? (
            <>
              <div className="flex border-gray-400 items-center border-b pb-4">
                <button onClick={handleCTA}>
                  <RiArrowLeftSLine className="text-primary" size={40} />
                </button>
                <h2 className="heading-sm font-light">Set up Face ID</h2>
              </div>
              <div className="border-gray-400 border-b py-3 flex justify-between items-center">
                <div>
                  <p className="text-sm text-gray-500 mb-5">
                    To set up face ID, position your face in the frame and look
                    directly at the camera.
                  </p>
                  <button
                    onClick={handleModalOpen}
                    className="text-white bg-primary px-10 py-2 flex gap-3 items-center"
                  >
                    Get Started <RiExternalLinkLine />
                  </button>
                </div>
              </div>
              {modalOpen && (
                <FaceModal handleModalClose={handleModalClose} from={"setup"} />
              )}
            </>
          ) : (
            <>
              <h2 className="heading-sm font-light border-gray-400 border-b pb-4">
                Profile and Preferences
              </h2>
              <div className="border-gray-400 border-b py-3 flex justify-between items-center">
                <div>
                  <button className="text-primary" onClick={handleCTA}>
                    Set up Face ID
                  </button>
                  <p className="text-sm text-gray-500">
                    Make log-ins more secure and convenient with face
                    authentication
                  </p>
                </div>
                <RiArrowRightSLine className="text-primary" />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default SetUpScreen;
