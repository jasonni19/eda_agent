import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Object EDA",
  description: "Upload a CSV and receive a exploratory data analysis report.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
