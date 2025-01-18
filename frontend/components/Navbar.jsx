"use client";

import {
  RiCloseFill,
  RiLockLine,
  RiMenu2Line,
  RiMenuLine,
  RiUser2Fill,
} from "@remixicon/react";
import Image from "next/image";
import Link from "next/link";
import React from "react";

function Navbar() {
  return (
    <>
      <header className="bg-primary">
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
            <div className="flex items-center">
              <Link
                href="/dashboard"
                className="border-white border p-2 h-full md:h-auto md:px-5 bg-secondary flex flex-col items-center"
              >
                {/* <RiLockLine className="md:hidden" /> */}
                <RiUser2Fill className="md:hidden" />
                Profile
              </Link>
              <div className="nav__toggle p-3 md:hidden">
                <RiMenuLine color="white" />
              </div>
            </div>
          </div>
        </div>
        <nav className="nav__desktop hidden md:block py-4 bg-white">
          <div className="container">
            <div className="nav__items">
              <ul className="flex gap-5 justify-end text-primary">
                <li>
                  <Link href="/">Accounts</Link>
                </li>
                <li>
                  <Link href="/">Credit Cards</Link>
                </li>
                <li>
                  <Link href="/">Mortages</Link>
                </li>
                <li>
                  <Link href="/">Loans</Link>
                </li>
                <li>
                  <Link href="/">Investments</Link>
                </li>
                <li>
                  <Link href="/">Rewards</Link>
                </li>
                <li>
                  <Link href="/">Advice</Link>
                </li>
              </ul>
            </div>
          </div>
        </nav>
      </header>
    </>
  );
}

export default Navbar;
