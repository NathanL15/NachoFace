"use client";

import {
  RiArrowDownLine,
  RiArrowDownSLine,
  RiCloseFill,
  RiLockLine,
  RiMenu2Line,
  RiMenuLine,
  RiUser2Line,
  RiUserFill,
} from "@remixicon/react";
import Image from "next/image";
import Link from "next/link";
import React from "react";

function DashNav({ switchScreen, isSetUpScreen }) {
  return (
    <>
      <header className="bg-white">
        <div className="container">
          <div className="nav__data flex items-stretch justify-between">
            <Link href="/" className="nav__logo p-2 flex">
              <Image
                src="/rbc-logo-shield.svg"
                width={50}
                height={60}
                alt="Logo"
              />
              <span className="font-semibold text-white">Royal Bank</span>
            </Link>
            <div className="flex items-center gap-5">
              <button className="flex gap-3 text-primary">
                <RiUserFill />
                <span className="hidden md:block">Richard Robert</span>{" "}
                <RiArrowDownSLine />
              </button>
              <Link
                href="/sign-in"
                className="border-white border p-2 h-full md:h-auto md:px-5 bg-secondary flex flex-col items-center"
              >
                <RiLockLine className="md:hidden" />
                Sign Out
              </Link>
              <div className="nav__toggle p-3 md:hidden">
                <RiMenuLine className="text-primary" />
              </div>
            </div>
          </div>
        </div>
        <nav className="nav__desktop md:block bg-primary">
          <div className="container">
            <div className="nav__items">
              <ul className="flex gap-5 justify-start text-white">
                <li className="py-4">
                  <Link href="/">Products & Services</Link>
                </li>
                <li className="py-4 border-b-4 border-secondary">
                  <Link href="/">My Accounts</Link>
                </li>
                <li className="py-4">
                  <Link href="/">Customer Service</Link>
                </li>
              </ul>
            </div>
          </div>
        </nav>
        <nav className="nav__desktop md:block bg-gray">
          <div className="container">
            <div className="nav__items">
              <ul className="flex gap-5 justify-start text-gray-600">
                <li
                  className={`py-4 border-primary ${
                    isSetUpScreen ? "" : "border-b-4"
                  }`}
                >
                  <button onClick={switchScreen}>Account Summary</button>
                </li>
                <li
                  className={`py-4 border-primary ${
                    isSetUpScreen ? "border-b-4" : ""
                  }`}
                >
                  <button onClick={switchScreen}>
                    Profile & Account Settings
                  </button>
                </li>
              </ul>
            </div>
          </div>
        </nav>
      </header>
    </>
  );
}

export default DashNav;
