import Link from "next/link";
import React from "react";

function SummaryScreen() {
  return (
    <section>
      <div className="container py-10">
        <h2 className="heading-sm mb-10">Welcome, Richard</h2>
        <div className="tabs border-gray-500 border-b">
          <button className="tabs__item heading-xs py-4 border-primary border-b-4">
            Bank Accounts
          </button>
        </div>
        <table className="table-auto dashboard-table w-full">
          <tbody>
            <tr>
              <td>
                <p>
                  <Link className="text-primary" href="/">
                    RBC Day to Day Banking
                  </Link>{" "}
                  <span className="text-gray-400 text-xs">
                    Chequing 00002-1234567
                  </span>
                </p>
              </td>
              <td>$650</td>
            </tr>
            <tr>
              <td>
                <Link className="text-primary" href="/">
                  RBC High Interest eSavings
                </Link>{" "}
                <span className="text-gray-400 text-xs">
                  Chequing 00002-1234567
                </span>
              </td>
              <td>$1050</td>
            </tr>
            <tr className="bg-gray-200">
              <td>
                <strong className="text-gray-500">Total</strong>
              </td>
              <td>
                <strong>$1700</strong>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default SummaryScreen;
