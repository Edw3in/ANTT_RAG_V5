import React from "react";

type ChipProps = {
  label: string;
};

export function Chip({ label }: ChipProps) {
  return <span style={styles.chip}>{label}</span>;
}

const styles: Record<string, React.CSSProperties> = {
  chip: {
    display: "inline-flex",
    alignItems: "center",
    padding: "4px 10px",
    borderRadius: 999,
    fontSize: 12,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(255,255,255,0.04)",
    opacity: 0.9,
    whiteSpace: "nowrap",
  },
};
