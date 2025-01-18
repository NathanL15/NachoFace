"use client";

import FaceModal from "@/components/FaceModal";
import { RiCheckLine } from "@remixicon/react";
import Image from "next/image";
import Link from "next/link";
import React, { useState } from "react";

function Page() {
  const [modalOpen, setModalOpen] = useState(false);

  const handleModalClose = () => {
    setModalOpen(false);
  };

  const handleModalOpen = (e) => {
    e.preventDefault();
    setModalOpen(true);
  };

  return (
    <>
      <section className="h-screen sign-in">
        <div className="grid grid-cols-1 md:grid-cols-2 h-full">
          <div className="sign-in__logo flex flex-col gap-3 font-semibold items-center justify-center">
            <Image
              src="/rbc-logo-shield.svg"
              width={50}
              height={70}
              alt="Logo"
            />
            <p className="text-white">Secure Sign-in</p>
          </div>
          <div className="sign-in__form bg-white h-full px-5 md:col-start-2 md:col-end-3">
            <form className="max-w-md flex flex-col gap-3 py-12 mx-auto">
              <div className="flex flex-col">
                <label className="mb-2" htmlFor="card">
                  Client Card or Username
                </label>
                <input
                  type="text"
                  id="card"
                  className="p-2 outline-primary border-gray-600 border"
                />
              </div>
              <div className="flex items-center">
                <div className="relative">
                  <input
                    type="checkbox"
                    id="saveCard"
                    className="w-8 h-8 border-gray-600 border appearance-none checked:bg-blue-500 checked:border-blue-500"
                  />
                  <RiCheckLine className="checkbox-fill" color="white" />
                </div>
                <label htmlFor="saveCard" className="ml-2 text-gray-700">
                  Save client card or username
                </label>
              </div>
              <button className="md:px-10 p-3 bg-white text-primary border-primary border">
                Next
              </button>
              <button
                onClick={(e) => {
                  handleModalOpen(e);
                }}
                className="md:px-10 p-3 bg-primary text-white font-bold"
              >
                Log-in with Face ID
              </button>
              <div className="flex flex-col gap-2">
                <Link className="text-primary" href="/">
                  Recover Your Username
                </Link>
                <Link className="text-primary" href="/">
                  Enrol in Online Banking
                </Link>
              </div>
            </form>
          </div>
        </div>
      </section>
      {modalOpen && <FaceModal handleModalClose={handleModalClose} />}
    </>
  );
}

export default Page;
