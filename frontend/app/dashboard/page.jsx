"use client";

import DashNav from "@/components/DashNav";
import SetUpScreen from "@/components/SetUpScreen";
import SummaryScreen from "@/components/SummaryScreen";
import React, { useState } from "react";

function Page() {
  const [isSetUpScreen, setUpScreen] = useState(false);

  const switchScreen = () => {
    setUpScreen(!isSetUpScreen);
  };

  return (
    <>
      <DashNav switchScreen={switchScreen} isSetUpScreen={isSetUpScreen} />
      {isSetUpScreen ? <SetUpScreen /> : <SummaryScreen />}
    </>
  );
}

export default Page;
