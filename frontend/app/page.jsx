import Navbar from "@/components/Navbar";

export default function Home() {
  return (
    <>
      <Navbar />
      <div className="home-banner h-[calc(100vh-74px)] md:h-[calc(100vh-130px)] flex items-center bg-gradient  text-white">
        <div className="container px-5">
          <div className="text">
            <span className="inline-block text-black font-bold p-1 rounded-md bg-secondary mb-3">
              RBC Chequing Account Offer
            </span>
            <h1 className="heading-md font-semibold max-w-3xl mb-3">
              Open an eligible RBC chequing account{" "}
              <span className="text-secondary">and get $450</span>
            </h1>
            <p className="mb-5">
              Offer Ends February 10, 2025. Conditions apply.
            </p>
            <button className="w-full md:px-10 md:w-auto p-3 bg-white text-primary text-bold">
              View Offer Details
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
