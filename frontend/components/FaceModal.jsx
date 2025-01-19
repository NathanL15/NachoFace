"use client";

import React, { useState } from "react";
import Camera from "./Camera";

export default function FaceModal({ handleModalClose }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFramesCaptured = async (frames) => {
    setLoading(true);
    try {
      const response = await fetch("/api/face-id", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ mode: "login", frames }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error during face recognition:", error);
      setResult({ error: "An error occurred. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white p-5 rounded shadow-lg w-full max-w-md">
        <h2 className="text-lg font-bold mb-4">Log In with Face ID</h2>
        <Camera onFramesCaptured={handleFramesCaptured} />
        <button
          onClick={handleModalClose}
          className="bg-gray-500 text-white px-4 py-2 rounded mt-4"
        >
          Close
        </button>
        {loading && <p className="text-gray-500 mt-2">Processing...</p>}
        {result && (
          <div className="mt-4">
            {result.error ? (
              <p className="text-red-500">{result.error}</p>
            ) : result.match ? (
              <p className="text-green-500">Login successful!</p>
            ) : (
              <p className="text-red-500">No match found.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
