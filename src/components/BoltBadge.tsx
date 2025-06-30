import React from "react";
import boltBadge from "../assets/bolt-badge-black.svg";

const badgeStyle: React.CSSProperties = {
  position: "fixed",
  left: 24,
  bottom: 24,
  zIndex: 50,
  width: 64,
  height: 64,
  minWidth: 48,
  minHeight: 48,
  maxWidth: "20vw",
  maxHeight: "20vw",
  cursor: "pointer",
  boxShadow: "0 2px 12px rgba(0,0,0,0.12)",
};

export default function BoltBadge() {
  return (
    <a
      href="https://bolt.new/"
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Built with Bolt.new"
      style={badgeStyle}
    >
      <img
        src={boltBadge}
        alt="Bolt.new badge"
        style={{ width: "100%", height: "100%" }}
      />
    </a>
  );
} 