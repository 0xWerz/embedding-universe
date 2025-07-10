import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Embedding Universe - Interactive Semantic Network Visualization",
  description:
    "Explore how AI understands language with an interactive semantic network. Add words and discover their connections through real-time AI embeddings. Browser-based, no server required.",

  keywords: [
    "AI embeddings",
    "semantic network",
    "language visualization",
    "transformers.js",
    "word relationships",
    "NLP visualization",
    "machine learning",
    "semantic similarity",
    "interactive AI",
    "browser AI",
    "word embeddings",
    "semantic space",
    "AI education",
    "language understanding",
    "cosine similarity",
  ],

  authors: [{ name: "Embedding Universe" }],
  creator: "Embedding Universe",
  publisher: "Embedding Universe",

  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },

  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL || "https://embedding-universe.com"
  ),

  alternates: {
    canonical: "/",
  },

  openGraph: {
    title: "Embedding Universe - Interactive Semantic Network Visualization",
    description:
      "Explore how AI understands language with an interactive semantic network. Add words and discover their connections through real-time AI embeddings.",
    url: "/",
    siteName: "Embedding Universe",
    type: "website",
    locale: "en_US",
    images: [
      {
        url: "/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "Embedding Universe - Interactive semantic network with connected word bubbles",
      },
    ],
  },

  twitter: {
    card: "summary_large_image",
    title: "Embedding Universe - Interactive Semantic Network Visualization",
    description:
      "Explore how AI understands language with an interactive semantic network. Browser-based AI, no server required.",
    images: ["/og-image.jpg"],
    creator: "@embedding_universe",
  },

  robots: {
    index: true,
    follow: true,
    nocache: true,
    googleBot: {
      index: true,
      follow: true,
      noimageindex: false,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },

  viewport: {
    width: "device-width",
    initialScale: 1,
    maximumScale: 5,
    userScalable: true,
  },

  verification: {
    google: process.env.GOOGLE_SITE_VERIFICATION,
  },

  category: "technology",

  other: {
    "theme-color": "#000000",
    "color-scheme": "dark light",
    "apple-mobile-web-app-capable": "yes",
    "apple-mobile-web-app-status-bar-style": "black-translucent",
    "apple-mobile-web-app-title": "Embedding Universe",
    "application-name": "Embedding Universe",
    "msapplication-TileColor": "#000000",
    "msapplication-config": "/browserconfig.xml",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* Additional SEO and branding meta tags */}
        <link rel="icon" href="/favicon.ico" />
        <link
          rel="apple-touch-icon"
          sizes="180x180"
          href="/apple-touch-icon.png"
        />
        <link
          rel="icon"
          type="image/png"
          sizes="32x32"
          href="/favicon-32x32.png"
        />
        <link
          rel="icon"
          type="image/png"
          sizes="16x16"
          href="/favicon-16x16.png"
        />
        <link rel="manifest" href="/site.webmanifest" />
        <link rel="mask-icon" href="/safari-pinned-tab.svg" color="#8b5cf6" />

        {/* Structured Data */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebApplication",
              name: "Embedding Universe",
              description:
                "Interactive semantic network visualization tool that helps users understand how AI models perceive language relationships using real-time embeddings.",
              url:
                process.env.NEXT_PUBLIC_SITE_URL ||
                "https://embedding-universe.com",
              applicationCategory: "EducationalApplication",
              operatingSystem: "Web Browser",
              browserRequirements:
                "Requires JavaScript and WebAssembly support",
              offers: {
                "@type": "Offer",
                price: "0",
                priceCurrency: "USD",
              },
              creator: {
                "@type": "Organization",
                name: "Embedding Universe",
              },
              featureList: [
                "Interactive semantic network visualization",
                "Real-time AI embeddings",
                "Browser-based AI processing",
                "No server required",
                "Educational tool for understanding AI language models",
                "Drag and zoom interface",
                "Live word connections",
              ],
              screenshot: "/og-image.jpg",
            }),
          }}
        />

        {/* Preload critical fonts */}
        <link
          rel="preload"
          href="/_next/static/media/geist-sans.woff2"
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
